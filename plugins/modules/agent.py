import random
from math import cos, exp, isnan, sin

# import numba as nb
import numpy as np
import torch
import torch.nn as nn
# Import the global bluesky objects. Uncomment the ones you need
from bluesky.tools.aero import ft
from bluesky.tools.geo import latlondist, nm

from modules.ppo import PPO


################
# HYPER PARAMS #
################


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dist(pos, pos2):
    return latlondist(pos[0], pos[1], pos2[0], pos2[1])/nm

# trk,alt, tas, vs,x,y,z,dist


def get_nearest_n(dist_matrix, _id, traf, local_traf, n=4):
    closest = []
    new_dist_traf = []
    new_local_traf = []
    if len(dist_matrix) > 0:
        for i, dist in enumerate(dist_matrix):
            if dist <= 100:
                new_dist_traf.append(dist)
                new_local_traf.append(local_traf[i])

    new_dist = new_dist_traf.copy()
    new_dist.sort()
    new_local = []
    for i in new_dist:
        new_local.append(new_local_traf[new_dist_traf.index(i)])

    if new_dist:
        for i in range(0, min(len(new_dist), n)):
            lat = traf.lat[new_local[i]]
            lon = traf.lon[new_local[i]]

            x = cos(lat)*cos(lon)
            y = cos(lat)*sin(lon)
            z = sin(lat)

            closest.append([traf.trk[new_local[i]], traf.alt[new_local[i]]/ft,
                            traf.tas[new_local[i]], traf.vs[new_local[i]], lat, lon, new_dist[i]])
    closest = closest
    for i in range(4-len(closest)):
        closest.append([0, 0, 0, 0, 0, 0, 0])

    return closest


def get_distance_matrix_ac(traf, _id, local_traf):
    idx = traf.id2idx(_id)
    distances = []

    for ac in local_traf:
        distances.append(latlondist(
            traf.lat[idx], traf.lon[idx], traf.lat[ac], traf.lon[ac])/nm)

    return distances

# Get the nearest aircraft distance


def nearest_ac(dist_matrix, _id, traf):
    idx = traf.id2idx(_id)

    closest = dist_matrix[0]
    close = closest[6]
    this_alt = traf.alt[idx]/ft
    close_alt = closest[1]
    alt_separations = abs(this_alt - close_alt)

    if close == 0 and alt_separations == this_alt:
        return (-1, -1)
    else:
        return close, alt_separations


class Agent():
    def __init__(self, state_size, action_size, action_std, lr, betas, gamma, K_epochs, eps_clip, terminal_dist=15, minsep=5, minvsep=2000, PATH=None):
        self.terminal_dist = terminal_dist
        self.minsep = minsep
        self.minvsep = minvsep

        self.ppo = PPO(state_size, action_size, action_std,
                       lr, betas, gamma, K_epochs, eps_clip)

        self.min_trk = 0
        self.max_trk = 360
        self.max_alt = 36000
        self.min_alt = 22000
        self.max_tas = 1
        self.min_tas = 0
        self.max_vs = 1
        self.min_vs = 0
        # self.max_x = 1
        # self.min_x = 0
        # self.max_y = 1
        # self.min_y = 0
        # self.max_z = 1
        # self.min_z = 0
        self.max_g_dist = 1
        self.min_g_dist = 0
        self.max_dist = 1
        self.min_dist = 0

        self.max_lat = 0.0001
        self.min_lat = -0.0001

        self.max_lon = 0.0001
        self.min_lon = -0.0001

        try:
            self.ppo.load_model(PATH)
        except:
            print("Unable to load model")

    def update_PPO(self, memory):
        self.ppo.update(memory)

    def save(self, PATH):
        self.ppo.save_model(PATH)

    def update(self, traf, traffic, memory):
        actions = []
        terminal = []
        rewards = []
        for _id in traf.id:
            ac_sector = traffic.get_in_sectors(_id, traf)
            sector_ac = []
            location = traf.id2idx(_id)

            for sector in ac_sector:
                sector_ac += (traffic.get_sector_indexes(sector, _id, traf))

            sector_ac = list(dict.fromkeys(sector_ac))

            nearest_n = get_nearest_n(get_distance_matrix_ac(
                traf, _id, sector_ac), _id, traf, sector_ac)

            nearest_one = nearest_ac(nearest_n, _id, traf)

            state = self.get_state(traf, _id, nearest_n, traffic.routes)

            _action = self.ppo.select_action(state, memory)
            action = np.argmax(_action)

            lat = traf.lat[location]
            lon = traf.lon[location]
            g_dist = get_dist([lat, lon],
                              traffic.routes[_id][-1])

            t_type = self.is_terminal(
                traf, _id, traffic.routes[_id], nearest_one)

            reward = self.get_reward(
                _id, traf, nearest_one, action, g_dist, t_type)

            rewards.append(reward)

            memory.rewards.append(reward)
            memory.is_terminals.append(t_type != 0)

            if not t_type == 0:
                terminal.append((_id, t_type))

            actions.append(action)

        return terminal, actions, np.mean(rewards)

    def normalise_self(self, self_state):
        self.max_tas = max(self_state[2], self.max_tas)
        self.min_tas = min(self_state[2], self.min_tas)

        self.max_vs = max(self_state[3], self.max_vs)
        self.min_vs = min(self_state[3], self.min_vs)

        # self.max_x = max(self_state[4], self.max_x)
        # self.min_x = min(self_state[4], self.min_x)
        # self.max_y = max(self_state[5], self.max_y)
        # self.min_y = min(self_state[5], self.min_y)
        # self.max_z = max(self_state[6], self.max_z)
        # self.min_z = min(self_state[6], self.min_z)
        self.max_g_dist = max(self_state[6], self.max_g_dist)
        self.min_g_dist = min(self_state[6], self.min_g_dist)
        self.max_lat = max(self_state[4], self.max_lat)
        self.min_lat = min(self_state[4], self.min_lat)
        self.max_lon = max(self_state[5], self.max_lon)
        self.min_lon = min(self_state[5], self.min_lon)

        self_state[0] = ((self_state[0]-self.min_trk) /
                         (self.max_trk-self.min_trk))

        self_state[1] = ((self_state[1]-self.min_alt) /
                         (self.max_alt-self.min_alt))

        self_state[2] = ((self_state[2]-self.min_tas) /
                         (self.max_tas-self.min_tas))

        self_state[3] = ((self_state[3]-self.min_vs) /
                         (self.max_vs-self.min_vs))

        self_state[4] = ((self_state[4]-self.min_lat) /
                         (self.max_lat-self.min_lat))

        self_state[5] = ((self_state[5]-self.min_lon) /
                         (self.max_lon-self.min_lon))

        # self_state[6] = ((self_state[6]-self.min_z) /
        #                  (self.max_z-self.min_z))

        self_state[6] = ((self_state[6]-self.min_g_dist) /
                         (self.max_g_dist-self.min_g_dist))

        return self_state

    def normalise_state(self, n_closest):
        final_state = []
        for state in n_closest:
            if not np.sum(state) == 0:
                t_state = []
                self.max_tas = max(state[2], self.max_tas)
                self.min_tas = min(state[2], self.min_tas)

                self.max_vs = max(state[3], self.max_vs)
                self.min_vs = min(state[3], self.min_vs)

                self.max_dist = max(state[6], self.max_dist)
                self.min_dist = min(state[6], self.min_dist)

                # self.max_x = max(self_state[4], self.max_x)
                # self.min_x = min(self_state[4], self.min_x)
                # self.max_y = max(self_state[5], self.max_y)
                # self.min_y = min(self_state[5], self.min_y)
                # self.max_z = max(self_state[6], self.max_z)
                # self.min_z = min(self_state[6], self.min_z)

                self.max_lat = max(state[4], self.max_lat)
                self.min_lat = min(state[4], self.min_lat)
                self.max_lon = max(state[5], self.max_lon)
                self.min_lon = min(state[5], self.min_lon)

                t_state.append((state[0]-self.min_trk) /
                               (self.max_trk-self.min_trk))

                t_state.append((state[1]-self.min_alt) /
                               (self.max_alt-self.min_alt))

                t_state.append((state[2]-self.min_tas) /
                               (self.max_tas-self.min_tas))

                t_state.append((state[3]-self.min_vs) /
                               (self.max_vs-self.min_vs))

                t_state.append((state[4]-self.min_lat) /
                               (self.max_lat-self.min_lat))

                t_state.append((state[5]-self.min_lon) /
                               (self.max_lon-self.min_lon))

                # t_state.append((state[6]-self.min_z) /
                #                  (self.max_z-self.min_z))

                t_state.append((state[6]-self.min_dist) /
                               (self.max_g_dist-self.min_dist))

                final_state.append(t_state)
            else:
                final_state.append(state)

        return final_state

    def get_reward(self, _id, traf, nearest_one, action, g_dist, t_type):
        idx = traf.id2idx(_id)

        if t_type == 2:
            return -1000

        l_sep, v_sep = nearest_one

        reward = 0

        # Get reward for distance

        if l_sep <= 50 and l_sep > 0:
            reward -= (1-(l_sep/50))*5

            if v_sep <= 2500:
                reward -= (1-min((v_sep/2500), 1))*10

        if reward == 0 and not action == 0:
            reward -= 10

        return reward

    def get_state(self, traf, _id, nearest_n, routes):

        idx = traf.id2idx(_id)

        lat = traf.lat[idx]
        lon = traf.lon[idx]
        g_dist = get_dist([lat, lon],
                          routes[_id][-1])

        # x = cos(lat)*cos(lon)
        # y = cos(lat)*sin(lon)
        # z = sin(lat)
        n_dist, _ = nearest_ac(nearest_n, _id, traf)
        # state = np.array([self.normalise_self([traf.trk[idx], traf.alt[idx]/ft, traf.tas[idx], traf.vs[idx],
        #                                        x, y, z, g_dist])])
        state = np.array([self.normalise_self([traf.trk[idx], traf.alt[idx]/ft, traf.tas[idx], traf.vs[idx],
                                               lat, lon, g_dist])]).flatten()
        nearest_n = np.array(self.normalise_state(nearest_n)).flatten()
        state = np.concatenate(
            ((state), (nearest_n)))

        return state

    def is_terminal(self, traf, _id, route, nearest_one):
        idx = traf.id2idx(_id)

        sep, vsep = nearest_one

        if (sep <= self.minsep and vsep <= self.minvsep) and (not sep < 0 and not vsep < 0):
            return 2

        pos = [traf.lat[idx], traf.lon[idx]]
        dest = route[-1]

        if get_dist(pos, dest) <= self.terminal_dist:
            return 1

        return 0
