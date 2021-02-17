""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
import datetime
import time
from random import randint

# Import the global bluesky objects. Uncomment the ones you need
import bluesky
import numpy as np
from bluesky import core, stack, traf  # , settings, navdb, sim, scr, tools
from bluesky.tools import geo
from bluesky.tools.aero import ft, kts
from bluesky.tools.geo import latlondist, nm

from modules.agent import Agent
from modules.airspace import Airspace
from modules.memory import Memory
from modules.sectors import load_sectors
from modules.traffic import Traffic

EPOCHS = 2500
MAX_AC = 40
STATE_SHAPE = 5


# PATH = "models/-7781194074839573161-BestModel.md5"
PATH = "models/" + \
    str(hash(datetime.datetime.now()))+"-BestModel.md5"

# Initialization function of your plugin. Do not change the name of this
# function, as it is the way BlueSky recognises this file as a plugin.


def init_plugin():
    ''' Plugin initialisation function. '''
    # Instantiate our example entity
    example = ATC()

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
        self.initilized = False

    # Functions that need to be called periodically can be indicated to BlueSky
    # with the timed_function decorator

    def init(self):
        # Load the sector bounds
        self.sectors = sectors = load_sectors(
            sector_path="sectors/case_c.json")
        self.airspace = Airspace(path="nodes/case_c.json")
        self.traffic = Traffic(max_ac=MAX_AC, network=self.airspace)
        self.memory = Memory()
        self.agent = Agent(STATE_SHAPE, 5,
                           5)
        self.max_agents = 0

        self.epoch_counter = 0
        self.success = 0
        self.fail = 0

        self.all_success = []
        self.all_fail = []
        self.mean_success = 0
        self.all_mean_success, self.best = 0, 0
        self.mean_rewards = []

        self.start = None
        self.stop = None

        self.actions = np.array([0, -2000, 2000, -20, 20])

        print("ATC: READY")
        string = "=================================\n   UPDATE: RUNNING EPOCH {}\n=================================\n".format(
            self.format_epoch())
        self.print_all(string)

        self.initilized = True

    @core.timed_function(name='atc', dt=12)
    def update(self):
        # Initilize env if not already
        if not self.initilized:
            self.init()

        # Start the epoch timer
        if not self.start:
            self.start = time.perf_counter()

        # Run the spawn command, this creates aircraft in the scenario.
        self.traffic.spawn()

        # Update the sectors each aircraft belongs to
        self.traffic.get_sectors(self.sectors, traf)

        terminal_ac = np.zeros(len(traf.id), dtype=int)
        terminal_id = []

        active_sectors = self.traffic.get_active(traf)

        # Generate a full distancematrix between each aircraft
        full_dist_matrix = self.get_dist_martix()

        # Loop through and get terminal aircraft
        for i in range(len(traf.id)):
            T = 0
            for x in range(len(self.traffic.traf_in_sectors)):
                valid = self.traffic.traf_in_sectors[x, i]
                # Run if AC is in zone, this prevents collisions in uncontrolled zones
                if valid:
                    T = self.agent.terminal(
                        traf, i, self.get_nearest_ac(i, full_dist_matrix), self.traffic, self.memory)
                    break
                # When AC is not in a zone
                else:
                    T = self.agent.terminal(
                        traf, i, None, self.traffic, self.memory)

            # Add T type to terminals
            if not T == 0:
                terminal_ac[i] = True
                terminal_id.append([traf.id[i], T])

            # try:
            #     call_sig = traf.id[i]
            #     self.agent.store(self.memory.previous_observation[call_sig], self.memory.previous_action[call_sig], [np.zeros(
            #         self.memory.previous_observation[call_sig][0].shape), (self.memory.previous_observation[call_sig][1].shape)], traf, call_sig, T)
            # except Exception as e:
            #     print(f'ERROR: ', e)

        # Remove all treminal aircraft
        self.handle_terminals(terminal_id)

        # See if all aircraft for an epoch have been created, i.e. epoch is finished
        if self.traffic.check_done():
            # Reset the environment
            self.epoch_reset()
            return

        if not len(traf.id) == 0:
            self.next_action = {}

            # lat,lon,alt,tas,trk,vs,ax
            state = np.zeros((len(traf.id), 7))

            non_T_ids = np.array(traf.id)[terminal_ac != 1]

            indexes = np.array([int(x[3:]) for x in traf.id])

            state[:, 0] = traf.lat
            state[:, 1] = traf.lon
            state[:, 2] = traf.alt
            state[:, 3] = traf.tas
            state[:, 4] = traf.trk
            state[:, 5] = traf.vs
            state[:, 6] = traf.ax

            normal_state, context = self.get_normals_states(
                state, next_action, state[0].shape[0], terminal_ac, full_dist_matrix, active_sectors)

    def get_nearest_ac(self, _id, dist_matrix):
        row = dist_matrix[:, _id]
        close = 10e+25
        alt_sep = 0

        for i, dist in enumerate(row):
            if i == _id:
                continue

            if dist < close:
                close = dist
                this_alt = traf.alt[_id]
                close_alt = traf.alt[i]
                alt_sep = abs(this_alt - close_alt)

        return close, alt_sep

    def get_dist_martix(self):
        size = traf.lat.shape[0]
        return geo.latlondist_matrix(np.repeat(traf.lat, size), np.repeat(
            traf.lon, size), np.tile(traf.lat, size), np.tile(traf.lon, size)).reshape(size, size)

    def handle_terminals(self, terminals):
        for _id, T in terminals:
            stack.stack("DEL {}".format(_id))
            self.traffic.active -= 1

            if T == 1:
                self.fail += 1
            else:
                self.success += 1

    def get_normals_states(self, state, next_action, no_states, terminal, distancematrix, sectors):
        number_of_aircraft = traf.lat.shape[0]

        normal_state = np.zeros((len(terminal[terminal != 1]), no_states))

        size = traf.lat.shape[0]
        index = np.arange(size).reshape(-1, 1)

        sort = np.array(np.argsort(distancematrix, axis=1))

        total_closest_states = []
        routecount = 0

        self.max_agents = 1

        count = 0

        for i in range(distancematrix.shape[0]):
            # We dont care about aircraft that are terminal (new actions dont hold gravity)
            # We also dont care about traffic that is not in a sector, as we assume that no collisions can occure outside of controlled space.
            if terminal[i] == 1 or len(sectors[i]) <= 0:
                continue

            normal_state[count, :] = agent.normalize_state(
                state[i], id_=traf.id[i])

    # Reset the environment for the next epoch
    def epoch_reset(self):
        # Reset the traffic creation
        self.traffic.reset()
        # Stop the timer
        self.stop = time.perf_counter()

        # Keep track of all success and failures
        self.all_success.append(self.success)
        self.all_fail.append(self.fail)

        # Calcuate total mean success
        self.all_mean_success = np.mean(self.all_success)

        # Calcuate rolling mean success
        if (self.epoch_counter+1) >= 50:
            self.mean_success = np.mean(self.all_success[-50:])

        # Train the model every 5 epochs
        if (self.epoch_counter+1) % 5 == 0:
            if self.mean_success >= self.best:
                print('----- Saving New Best Model -----')
                # self.agent.save(PATH)

            print('----- Training Model -----')
            # self.agent.update_PPO(self.memory)
            self.memory.clear_memory()

        # Get the best rolling mean
        self.best = max(self.mean_success, self.best)

        # -------- Printing Outputs --------
        string = "Epoch run in {:.2f} seconds".format(self.stop-self.start)
        self.print_all(string)
        string = "Success: {} | Fail: {} | Mean Success: {:.3f} | Mean Reward {:.2f} | (50) Mean Success Rolling {:.3f} | Best {:.3f}".format(
            self.success, self.fail, self.all_mean_success/MAX_AC, np.mean(self.mean_rewards), self.mean_success/MAX_AC, self.best/MAX_AC)
        self.print_all(string)

        if self.epoch_counter+1 >= EPOCHS:
            stack.stack("STOP")

        self.epoch_counter += 1
        string = "=================================\n   UPDATE: RUNNING EPOCH {}\n=================================\n".format(
            self.format_epoch())
        self.print_all(string)

        # Reset values
        self.success = 0
        self.fail = 0
        self.stop = None
        self.start = None
        self.mean_rewards = []

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
