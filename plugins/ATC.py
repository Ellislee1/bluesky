""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
import time
from random import randint

import bluesky
import numpy as np
from bluesky import core, stack, traf  # , settings, navdb, sim, scr, tools
from bluesky.tools import geo
from bluesky.tools.aero import ft, kts
from bluesky.tools.geo import latlondist, nm
from tensorflow import keras

from modules.sectors import Sector_Manager
from modules.routes import Route_Manager
from modules.traffic import Traffic_Manager
from modules.agent import Agent, get_goal_dist, get_dist, get_n_nodes
from modules.memory import Memory

####################
##  HYPER PARAMS  ##
####################
# Training
TRAIN = True
# Visual
VISUALIZE = True
# Files
ROUTES = "routes/routes_b2.json"
SECTORS = "sectors/sectors_b.2.json"
FILE = "NULL"
NEW_FILE = "case_a.3"
# Other
MAX_AC = 18
EPOCHS = 5000
# [spd,alt,hdg,vs,acc,d_goal,wp_id1,wp_id2]
STATE_SHAPE = 9
# ACTION_SHAPE = 3
# VALUE_SHAPE = 3
ACTION_SHAPE = 4
VALUE_SHAPE = 4
TIME_SEP = [20, 25, 30]
# ACTIONS = [0, 3000, -3000]
# ACTIONS = [-3000, 0, 3000, 0]
ACTIONS = [36000, 0, -28000, 0]

CONSTRAINTS = {
    "alt": {
        "min": 28000,
        "max": 36000
    },
    "cas": {
        "min": 250,
        "max": 350
    }
}
####################


# Initialization function of your plugin. Do not change the name of this
# function, as it is the way BlueSky recognises this file as a plugin.


def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity
    atc = ATC()

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'ATC',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',
    }

    # init_plugin() should always return a configuration dict.
    return config


# Entities in BlueSky are objects that are created only once (called singleton)
# which implement some traffic or other simulation functionality.
# To define an entity that ADDS functionality to BlueSky, create a class that
# inherits from bluesky.core.Entity.
# To replace existing functionality in BlueSky, inherit from the class that
# provides the original implementation (see for example the asas/eby plugin).
class ATC(core.Entity):
    ''' Example new entity object for BlueSky. '''

    def __init__(self):
        super().__init__()
        self.super_start = time.perf_counter()

        self.initilized = False

        self.epoch_counter = 0
        # [Success, Fail]
        self.results = np.zeros(2)

        self.all_success = []
        self.all_fail = []
        self.mean_success = 0
        self.all_mean_success, self.best = 0, 0
        self.mean_rewards = []
        self.epoch_actions = np.zeros(ACTION_SHAPE)

        self.start = None
        self.stop = None

        self.dist = [0, -1]
        self.spd = [0, -1]
        self.trk = [0, 360]
        self.vs = [0, -1]

        self.last_observation = {}
        self.last_reward_observation = {}
        self.previous_action = {}
        self.observation = {}

    def on_load(self):
        self.sector_manager = Sector_Manager(SECTORS)
        self.route_manager = Route_Manager(
            ROUTES, test_routes=VISUALIZE, draw_paths=VISUALIZE)
        self.traffic_manager = Traffic_Manager(
            max_ac=MAX_AC, times=TIME_SEP, max_spd=CONSTRAINTS[
                "cas"]["max"], min_spd=CONSTRAINTS["cas"]["min"],
            max_alt=32000, min_alt=32000, network=self.route_manager)

        self.memory = Memory()

        self.agent = Agent(state_size=STATE_SHAPE,
                           action_size=ACTION_SHAPE, value_size=VALUE_SHAPE)

        try:
            self.agent.load(path=FILE+"best.h5")
        except:
            try:
                self.agent.load(path=FILE+".h5")
            except:
                pass

        self.initilized = True

        print("ATC: READY")
        string = "=================================\n   UPDATE: RUNNING EPOCH {}\n=================================\n".format(
            self.format_epoch())
        self.print_all(string)

    # Functions that need to be called periodically can be indicated to BlueSky
    # with the timed_function decorator

    @core.timed_function(name='example', dt=12)
    def update(self):
        # Initilize system
        if not self.initilized:
            self.on_load()

        # Start epoch timer
        if not self.start:
            self.start = time.perf_counter()

        # Create aircraft
        self.traffic_manager.spawn()
        # Update Aircraft active sectors
        self.traffic_manager.update_active(self.sector_manager.system_sectors)

        # Generate a full distancematrix between each aircraft
        full_dist_matrix = self.get_dist_martix()

        # Get nearest ac in a matrix
        nearest_ac = self.get_nearest_ac(dist_matrix=full_dist_matrix)

        # Get goal distances for each aircraft
        g_distance = self.get_goal_distances()

        # Get an array of terminal aircraft
        terminal_ac, terminal_id = self.get_terminal(nearest_ac, g_distance)

        self.handle_terminal(terminal_id)

        if self.traffic_manager.check_done():
            self.epoch_reset()
            return

        if not TRAIN and (self.traffic_manager.total % 50 == 0):
            string = "Success: {} | Fail: {} | Mean Success: {:.3f}%".format(
                int(self.results[0]), int(self.results[1]), (self.results[0]/MAX_AC)*100)
            self.print_all(string)

        if len(traf.id) <= 0:
            return

        if not len(traf.id) == 0:
            policy, normal_state, normal_context = self.get_actions(
                terminal_ac, g_distance, full_dist_matrix)

            if len(policy) > 0:
                idx = 0
                new_actions = {}
                for i in range(len(traf.id)):
                    if terminal_ac[i] == 0 and len(self.traffic_manager.active_sectors[i]) > 0:
                        if not np.any(np.isnan(policy[idx])):
                            _id = traf.id[i]

                            if not _id in self.last_observation.keys():
                                self.last_observation[_id] = [
                                    normal_state[idx], normal_context[idx]]

                            action = np.random.choice(
                                ACTION_SHAPE, 1, p=policy[idx].flatten())[0]

                            # print(policy[idx], action)

                            self.epoch_actions[action] += 1

                            if not _id in self.observation.keys() and _id in self.previous_action.keys():
                                self.observation[_id] = [
                                    normal_state[idx], normal_context[idx]]

                                self.memory.store(
                                    _id, self.last_observation[_id], self.previous_action[_id], nearest_ac[idx])

                                self.last_observation[_id] = self.observation[_id]

                                del self.observation[_id]

                            self.perform_action(i, action)

                            new_actions[_id] = action

                        self.previous_action = new_actions

                        idx += 1

    # Act
    def get_actions(self, terminal_ac, g_dists, dist_matrix):
        ids = []
        new_actions = {}

        state = self.get_state()

        normal_state, normal_context = self.normalise_all(
            state, terminal_ac, g_dists, dist_matrix)

        policy = []
        if not len(normal_state) == 0:
            policy = self.agent.act(normal_state, normal_context)

        return policy, normal_state, normal_context

    # For an aircraft perform an action
    def perform_action(self, i, action):
        if action < 3:
            traf_alt = int(traf.alt[i]/ft)
            new_alt = int(round((traf_alt+ACTIONS[action])))

            alt = max(CONSTRAINTS["alt"]["min"], min(
                CONSTRAINTS["alt"]["max"], new_alt))

            # print(traf_alt, alt)

            stack.stack("{} alt {}".format(traf.id[i], alt))
        elif action == 4:
            traf_alt = traf.alt[i]/ft
            new_alt = int(round((traf_alt)))

    # Get the current state

    def get_state(self):
        state = np.zeros((len(traf.id), 6))

        start_ids, end_ids = self.get_all_nodes()

        state[:, 0] = traf.lat
        state[:, 1] = traf.lon
        state[:, 2] = traf.trk
        state[:, 3] = traf.alt
        state[:, 4] = traf.tas
        state[:, 5] = traf.vs

        return state

    # Get all nodes for each aircraft
    def get_all_nodes(self):
        start_ids = np.zeros(len(traf.id), dtype=int)
        end_ids = np.zeros(len(traf.id), dtype=int)

        for i in range(len(traf.id)):
            _id = traf.id[i]
            route = self.traffic_manager.routes[_id]
            start_ids[i] = np.argwhere(self.route_manager.idx_array ==
                                       route[0])
            end_ids[i] = np.argwhere(self.route_manager.idx_array ==
                                     route[-1])

        return start_ids, end_ids

    # Normalise the state and context
    def normalise_all(self, state, terminal_ac, g_dists, dist_matrix):
        normal_states = self.normalise_state(state, terminal_ac, g_dists)

        normal_context = []

        start_ids, end_ids = self.get_all_nodes()

        max_agents = 0
        for _id in traf.id:
            if terminal_ac[traf.id2idx(_id)] > 0 or len(self.traffic_manager.active_sectors[traf.id2idx(_id)]) <= 0:
                continue

            new_context = self.normalise_context(
                _id, terminal_ac, dist_matrix, start_ids, end_ids)

            max_agents = max(max_agents, len(new_context))

            if len(normal_context) == 0:
                normal_context = new_context
            else:
                normal_context = np.append(
                    keras.preprocessing.sequence.pad_sequences(normal_context, max_agents, dtype='float32'), keras.preprocessing.sequence.pad_sequences(new_context, max_agents, dtype='float32'), axis=0)

        if len(normal_context) == 0:
            normal_context = np.array([0, 0, 0, 0, 0, 0, 0]).reshape(1, 1, 7)

        # print(normal_states.shape, normal_context.shape)
        return normal_states, normal_context

    # Normalise the agent state only
    def normalise_state(self, state, terminal_ac, g_dists):
        total_active = 0

        for i in range(len(terminal_ac)):
            if terminal_ac[i] == 0 and len(self.traffic_manager.active_sectors[i]) > 0:
                total_active += 1

        normalised_state = np.zeros((total_active, STATE_SHAPE))

        count = 0
        for i in range(len(traf.id)):
            if terminal_ac[i] > 0 or len(self.traffic_manager.active_sectors[i]) <= 0:
                continue

            normalised_state[count, :] = self.normalise(
                state[i], 'state', traf.id[i], g_dist=g_dists[i])

            count += 1

        return normalised_state

    # Get and normalise context
    def normalise_context(self, _id, terminal_ac, dist_matrix, start_ids, end_ids):
        context = []
        idx = traf.id2idx(_id)

        distances = dist_matrix[:, idx]
        this_sectors = self.traffic_manager.active_sectors[idx]

        this_lat, this_lon = traf.lat[idx], traf.lon[idx]

        for i in range(len(distances)):
            # Ignore current aircraft
            if i == idx:
                continue

            if terminal_ac[i] > 0 or len(self.traffic_manager.active_sectors[i]) <= 0:
                continue

            sectors = self.traffic_manager.active_sectors[i]

            # Only care if the ac in a matching sector
            flag = False
            for x in sectors:
                if x in this_sectors:
                    flag = True

            if not flag:
                continue

            dist = get_dist([this_lat, this_lon], [traf.lat[i], traf.lon[i]])

            # Only care about visible distance aircraft
            if dist > 40:
                continue

            spd = traf.tas[i]
            alt = traf.alt[i]
            trk = traf.trk[i]
            vs = traf.vs[i]
            start_id = start_ids[i]
            end_id = end_ids[i]

            self.dist[1] = max(self.dist[1], dist)
            self.spd[1] = max(self.spd[1], spd)
            self.vs[1] = max(self.vs[1], vs)

            dist = dist/self.dist[1]
            spd = spd/self.spd[1]
            trk = trk/self.trk[1]
            alt = ((alt/ft)-CONSTRAINTS["alt"]["min"]) / \
                (CONSTRAINTS["alt"]["max"]-CONSTRAINTS["alt"]["min"])

            vs = 0
            if not vs == 0:
                vs = vs/self.vs[1]

            n_nodes, dist2next = get_n_nodes(
                traf.id[i], self.traffic_manager, self.route_manager)

            self.dist[1] = max(self.dist[1], dist2next)
            dist2next = dist2next/self.dist[1]

            if len(context) == 0:
                context = np.array(
                    [spd, alt, trk, vs, dist, dist2next, n_nodes[0], n_nodes[1], n_nodes[2]]).reshape(1, 1, 9)
            else:
                context = np.append(context, np.array(
                    [spd, alt, trk, vs, dist, dist2next, n_nodes[0], n_nodes[1], n_nodes[2]]).reshape(1, 1, 9), axis=1)

        if len(context) == 0:
            context = np.zeros(9).reshape(1, 1, 9)

        return context

    # perform normalisation
    def normalise(self, state, what, _id, g_dist=None):

        # Normalise the entire state
        if what == 'state':
            if not g_dist:
                raise Exception(
                    "For normalising a state please pass the distance to the goal.")

            self.dist[1] = max(self.dist[1], g_dist)
            self.spd[1] = max(self.spd[1], state[4])
            self.vs[1] = max(self.vs[1], state[5])

            dist = g_dist/self.dist[1]
            spd = state[4]/self.spd[1]
            trk = state[2]/self.trk[1]
            alt = ((state[3]/ft)-CONSTRAINTS["alt"]["min"]) / \
                (CONSTRAINTS["alt"]["max"]-CONSTRAINTS["alt"]["min"])

            vs = 0
            if not state[5] == 0:
                vs = state[5]/self.vs[1]

            n_nodes, dist2next = get_n_nodes(
                _id, self.traffic_manager, self.route_manager)

            self.dist[1] = max(self.dist[1], dist2next)
            dist2next = dist2next/self.dist[1]

            return np.array([spd, alt, trk, vs, dist, dist2next, n_nodes[0], n_nodes[1], n_nodes[2]])

    # Get the terminal aircraft
    def get_terminal(self, nearest_ac, g_dists):
        terminal_ac = np.zeros(len(traf.id), dtype=int)
        terminal_id = []

        # Loop through all aircraft
        for i in range(len(traf.id)):
            # Terminal state 0 = not terminal, 1 = collision, 2 = success
            T = 0

            # Only care about aircraft in a sector
            if len(self.traffic_manager.active_sectors[i]) > 0:
                close_ac = nearest_ac[i]
                n_ac_data = (close_ac[0], close_ac[1])

                # Get the terminal state
                T = self.agent.terminal(
                    i, n_ac_data, g_dists[i])

                # Only care about terminal aircraft
                if not T == 0:
                    # Update collision aircraft
                    if T == 1:
                        terminal_ac[i] = 1
                        terminal_ac[traf.id2idx(close_ac[2])] = 1
                    elif not terminal_ac[i] == 1:
                        terminal_ac[i] = 2

                    _id = traf.id[i]
                    self.memory.store(
                        _id, self.last_observation[_id], self.previous_action[_id], nearest_ac[i], T)

        for i in range(len(terminal_ac)):
            if terminal_ac[i] > 0:
                terminal_id.append([traf.id[i], terminal_ac[i]])

        return terminal_ac, terminal_id

    # Handle terminal aircraft
    def handle_terminal(self, terminal_id):
        for ac in terminal_id:
            stack.stack('DEL {}'.format(ac[0]))

            self.traffic_manager.active -= 1

            if ac[1] == 1:
                self.results[1] += 1
            elif ac[1] == 2:
                self.results[0] += 1

    # Generates a distance matrix of all aircraft in the system
    def get_dist_martix(self):
        size = traf.lat.shape[0]
        return geo.latlondist_matrix(np.repeat(traf.lat, size), np.repeat(
            traf.lon, size), np.tile(traf.lat, size), np.tile(traf.lon, size)).reshape(size, size)

    # Get the nearest aircraft to agents
    def get_nearest_ac(self, dist_matrix):
        nearest = []

        # Loop through all aircraft
        for i in range(len(traf.id)):
            a_alt = traf.alt[i]/ft

            ac_dists = dist_matrix[:, i]

            close = 10e+25
            alt_sep = 10e+25

            nearest_id = None

            # Loop through the row on the dist matrix
            for x in range(len(ac_dists)):
                # Ensure the aircraft is in controlled airspace and not the current aircraft
                if not x == i and len(self.traffic_manager.active_sectors[x]) > 0:

                    # See if it is closest and update
                    if ac_dists[x] < close:
                        close = float(ac_dists[x])
                        i_alt = traf.alt[x]/ft

                        alt_sep = abs(a_alt-i_alt)

                        nearest_id = traf.id[x]
            nearest.append([close, alt_sep, nearest_id])

        return np.array(nearest)

    # returns a matrix of distances to a goal
    def get_goal_distances(self):
        goal_ds = np.zeros(len(traf.id), dtype=float)

        for i in range(len(traf.id)):
            goal_ds[i] = get_goal_dist(
                traf.id[i], self.traffic_manager, self.route_manager)

        return goal_ds

    # Reset the environment for the next epoch
    def epoch_reset(self):
        # Reset the traffic creation
        self.traffic_manager.reset()

        # Keep track of all success and failures
        self.all_success.append(self.results[0])
        self.all_fail.append(self.results[1])

        # Calcuate total mean success
        self.all_mean_success = np.mean(self.all_success)

        # Calcuate rolling mean success
        if (self.epoch_counter+1) >= 50:
            self.mean_success = np.mean(self.all_success[-50:])

        if (self.epoch_counter+1) % 5 == 0:
            if self.mean_success > self.best:
                if TRAIN:
                    print('::::::: Saving Best ::::::')
                    self.agent.save(path=NEW_FILE+"best.h5")
                self.best = self.mean_success
            if TRAIN:
                print(':::::: Saving Model ::::::')
                self.agent.save(path=NEW_FILE+".h5")
                print(":::::::: Training ::::::::")
                self.agent.train(self.memory)
                print(":::::::: Complete ::::::::")

        temp = np.array([np.array(self.all_success), np.array(self.all_fail)])
        np.savetxt("Files/"+NEW_FILE+"_numpy.csv", temp, delimiter=',')

        # Stop the timer
        self.stop = time.perf_counter()
        # -------- Printing Outputs --------
        string = "Epoch run in {:.2f} seconds".format(self.stop-self.start)
        self.print_all(string)
        string = "Success: {} | Fail: {} | Mean Success: {:.3f}% | (50) Mean Success Rolling {:.3f}% | Best {:.3f}%".format(
            int(self.results[0]), int(self.results[1]), (self.all_mean_success/MAX_AC)*100, (self.mean_success/MAX_AC)*100, (self.best/MAX_AC)*100)
        self.print_all(string)
        string = "Actions -> Descend: {}, Hold Current: {}, Climb: {}, Maintain Climb: {}".format(
            self.epoch_actions[0], self.epoch_actions[1], self.epoch_actions[2], self.epoch_actions[3])
        # string = "Actions -> Descend: {}, Climb: {}".format(
        #     self.epoch_actions[1], self.epoch_actions[0])
        self.print_all(string)

        if self.epoch_counter+1 >= EPOCHS:
            super_stop = time.perf_counter()
            stack.stack("STOP")
            string = "::END:: Training {} episodes took {:.2f} hours".format(
                EPOCHS, ((super_stop-self.super_start)/60)/60)
            self.print_all(string)
            return

        self.epoch_counter += 1
        string = "=================================\n   UPDATE: RUNNING EPOCH {}\n=================================\n".format(
            self.format_epoch())
        self.print_all(string)

        # Reset values
        self.results = np.zeros(2)
        self.stop = None
        self.start = None
        self.mean_rewards = []
        self.epoch_actions = []
        self.epoch_actions = np.zeros(ACTION_SHAPE)

        self.previous_action = {}
        self.last_observation = {}
        self.observation = {}

    # Scripts for printing values
    def print_all(self, string):
        stack.stack(f'ECHO {string}')
        print(string)

    def format_epoch(self):
        epoch_string = ""

        if self.epoch_counter+1 < 10:
            epoch_string += "0"
        if self.epoch_counter+1 < 100:
            epoch_string += "0"
        if self.epoch_counter+1 < 1000:
            epoch_string += "0"
        if self.epoch_counter+1 < 10000:
            epoch_string += "0"

        epoch_string += str(self.epoch_counter+1)
        return epoch_string
