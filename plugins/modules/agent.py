import numpy as np
from bluesky.tools.aero import ft
from bluesky.tools.geo import latlondist, nm
import tensorflow as tf
from tensorflow import keras
from modules.ppo import PPO
from math import ceil

import numba as nb


HIDDEN_SIZE = 50
GAMMA = 0.9


@nb.njit()
# Discounted rewards
def discount(r, discounted_r, cum_r):
    for t in range(len(r) - 1, -1, -1):
        cum_r = r[t] + cum_r * GAMMA
        discounted_r[t] = cum_r
    return discounted_r

# Get the distance between two points.


def get_dist(pos1, pos2):
    return latlondist(pos1[0], pos1[1], pos2[0], pos2[1])/nm


# Get the route length to the goal
def get_goal_dist(_id, traf, traffic):
    idx = traf.id2idx(_id)
    active_waypoint = [traf.actwp.lat[idx], traf.actwp.lon[idx]]
    route = traffic.routes[_id]

    start = 0
    found = False
    min_dist = 9999
    i = 1
    while not found:
        dist = get_dist(active_waypoint, route[i])
        if dist > min_dist:
            found = True
        elif dist < min_dist:
            min_dist = dist
            start = i
        i += 1
        if i >= len(route):
            found = True

    d_goal = 0
    current = [traf.lat[idx], traf.lon[idx]]
    for i in range(start, len(route)):
        d_goal += get_dist(current, route[i])
        current = route[i]
    return d_goal

# Get a distance matrix


def get_distance_matrix_ac(traf, _id, local_traf):
    idx = traf.id2idx(_id)
    distances = []

    for ac in local_traf:
        distances.append(latlondist(
            traf.lat[idx], traf.lon[idx], traf.lat[ac], traf.lon[ac])/nm)

    return distances


# Get the nearest n aircraft
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
            alt = traf.alt[new_local[i]]/ft
            tas = traf.tas[new_local[i]]
            vs = traf.vs[new_local[i]]
            trk = traf.trk[new_local[i]]

            closest.append([lat, lon, alt, tas, vs, trk])
    closest = closest
    for i in range(5-len(closest)):
        closest.append([0, 0, 0, 0, 0, 0])

    return closest


# Get the nearest aircraft to the agent
def nearest_ac(dist_matrix, _id, traf):
    idx = traf.id2idx(_id)

    closest = dist_matrix[0]
    close = get_dist([traf.lat[idx], traf.lon[idx]], [
                     closest[0], closest[1]])
    this_alt = traf.alt[idx]/ft
    close_alt = closest[2]
    alt_separations = abs(this_alt - close_alt)

    if close == 0 and alt_separations == this_alt:
        return (10e+5, 10e+5)
    else:
        return close, alt_separations


class Agent:
    def __init__(self, statesize, actionsize, valuesize, num_intruders=5):
        self.num_intruders = num_intruders
        self.action_size = actionsize

        self.model = PPO(statesize, 5, actionsize, valuesize)

    def terminal(self, traf, i, nearest, traffic, memory):
        # get ac index in traffic array
        _id = traf.id[i]
        """
            0 = not terminal
            1 = collision
            2 = goal reached
        """
        if nearest:
            dist, alt = nearest
        else:
            dist, alt = 10e+11, 10e+11

        d_goal = get_goal_dist(_id, traf, traffic)
        memory.dist_goal[_id] = d_goal

        memory.dist_close[_id] = dist
        # Linear goal distance
        g_dist = get_dist([traf.lat[i], traf.lon[i]],
                          traffic.routes[_id][-1])
        # T = Terminal type
        T = self.is_terminal(dist, alt, g_dist)

        return T

    def is_terminal(self, distance, v_sep, d_goal):
        if distance <= 5 and v_sep/ft < 2000:
            return 1

        if d_goal <= 5:
            return 2

        return 0

    def act(self, state, context):
        context = context.reshape((state.shape[0], -1, 10))

        if context.shape[1] > self.num_intruders:
            context = context[:, -self.num_intruders:, :]
        elif context.shape[1] < self.num_intruders:
            context = keras.preprocessing.sequence.pad_sequences(
                context, self.num_intruders, dtype='float32')

        policy, value = self.model.estimator.predict({'input_state': state, 'input_context': context, 'empty': np.zeros(
            (state.shape[0], HIDDEN_SIZE))}, batch_size=state.shape[0])

        return policy, value

    def train(self, memory):
        """Grab samples from batch to train the network"""

        total_state = []
        total_reward = []
        total_A = []
        total_advantage = []
        total_context = []
        total_policy = []

        total_length = 0

        for transitions in memory.experience.values():
            episode_length = transitions['state'].shape[0]
            total_length += episode_length

            # state = transitions['state'].reshape((episode_length,self.state_size))
            state = transitions['state']
            context = transitions['context']
            reward = transitions['reward']
            done = transitions['done']
            action = transitions['action']

            discounted_r, cumul_r = np.zeros_like(reward), 0
            discounted_rewards = discount(reward, discounted_r, cumul_r)

            policy, values = self.model.estimator.predict(
                {'input_state': state, 'input_context': context, 'empty': np.zeros((len(state), HIDDEN_SIZE))}, batch_size=256)

            advantages = np.zeros((episode_length, self.action_size))

            index = np.arange(episode_length)
            advantages[index, action] = 1
            A = discounted_rewards - values[:, 0]

            if len(total_state) == 0:
                total_state = state
                if context.shape[1] == memory.max_agents:
                    total_context = context
                else:
                    total_context = keras.preprocessing.sequence.pad_sequences(
                        context, memory.max_agents, dtype='float32')
                total_reward = discounted_rewards
                total_A = A
                total_advantage = advantages
                total_policy = policy
            else:
                total_state = np.append(total_state, state, axis=0)
                if context.shape[1] == memory.max_agents:
                    total_context = np.append(total_context, context, axis=0)
                else:
                    total_context = np.append(total_context, keras.preprocessing.sequence.pad_sequences(
                        context, memory.max_agents, dtype='float32'), axis=0)
                total_reward = np.append(
                    total_reward, discounted_rewards, axis=0)
                total_A = np.append(total_A, A, axis=0)
                total_advantage = np.append(
                    total_advantage, advantages, axis=0)
                total_policy = np.append(total_policy, policy, axis=0)

        total_A = (total_A - total_A.mean())/(total_A.std() + 1e-8)

        self.model.model.fit({
            'input_state': total_state, 'input_context': total_context, 'empty': tf.zeros(shape=(total_length, HIDDEN_SIZE)), 'advantage': total_A, 'old_predictions': total_policy
        },
            {
                'policy_out': tf.cast(total_advantage, tf.float32), 'value_out': tf.cast(total_reward, tf.float32)
        }, shuffle=True, batch_size=total_state.shape[0], epochs=8, verbose=0, steps_per_epoch=10)
