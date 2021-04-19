from math import ceil

import numba as nb
import numpy as np
import tensorflow as tf
from bluesky import traf
from bluesky.tools.aero import ft
from bluesky.tools.geo import latlondist, nm
from tensorflow import keras

from modules.PPO import PPO


@nb.njit()
def discount_reward(r, discounted_r, cumul_r):
    """ Compute the gamma-discounted rewards over an episode
    """
    for t in range(len(r)-1, -1, -1):
        cumul_r = r[t] + cumul_r * 0.99
        discounted_r[t] = cumul_r
    return discounted_r


def get_next_node(_id, traffic, routes):
    idx = traf.id2idx(_id)
    active_waypoint = [traf.actwp.lat[idx], traf.actwp.lon[idx]]
    route = traffic.routes[_id]

    wpt = 0
    best_dist = 10e+25

    for i in range(1, len(route)):
        wpt_coords = routes.get_coords(route[i])

        dist = get_dist(active_waypoint, wpt_coords)

        if dist < best_dist:
            best_dist = dist
            wpt = i

    return wpt


def get_n_nodes(_id, traffic, routes, n=3):
    idx = traf.id2idx(_id)
    route = traffic.routes[_id]

    next_nodes = np.zeros(3)

    next_node = get_next_node(_id, traffic, routes)
    coords = routes.get_coords(route[next_node])

    next_nodes[0] = next_node/(len(routes.idx_array)-1)
    dist = get_dist([traf.lat[idx], traf.lon[idx]], coords)

    for i in range(1, n):
        if (next_node+1) < len(route):
            next_node += 1
            next_nodes[i] = next_node/(len(routes.idx_array)-1)
        else:
            next_nodes[i] = next_node/(len(routes.idx_array)-1)

    return next_nodes, dist


# Get the route length to the goal
def get_goal_dist(_id, traffic, routes):
    idx = traf.id2idx(_id)
    route_idx = int(_id[3:])
    route = traffic.routes[route_idx]
    route = int(route)

    return get_dist([traf.lat[idx], traf.lon[idx]], routes.all_routes[route]["points"][-1])


def get_dist(pos1, pos2):
    return latlondist(pos1[0], pos1[1], pos2[0], pos2[1])/nm


class Agent:
    def __init__(self, state_size, action_size, value_size, hidden_size=32):
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.value_size = value_size
        self.ppo = PPO(state_size=state_size,
                       action_size=action_size, value_size=value_size)
        self.model = self.ppo.build_model()

    # Find if the aircraft is terminal
    def terminal(self, i, nearest, g_dist):
        # get ac index in traffic array
        _id = traf.id[i]
        """
            0 = not terminal
            1 = collision
            2 = goal reached
        """

        dist, alt = nearest

        dist, alt = float(dist), float(alt)

        if dist <= 3 and alt < 2000:
            return 1

        if g_dist <= 10:
            return 2

        return 0

    # Get the actions
    def act(self, state, context):

        context = context.reshape((state.shape[0], -1, 6))
        if context.shape[1] > 5:
            context = context[:, -5:, :]
        if context.shape[1] < 5:
            context = tf.keras.preprocessing.sequence.pad_sequences(
                context, 5, dtype='float32')

        policy, value = self.ppo.predictor.predict({'input_states': state, 'context_input': context, 'empty': np.zeros(
            (state.shape[0], self.hidden_size))}, batch_size=state.shape[0])

        return policy

    def train(self, memory):
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

            state = transitions['state']
            context = transitions['context']
            reward = transitions['reward']
            done = transitions['done']
            action = transitions['action']

            discounted_r, cumul_r = np.zeros_like(reward), 0
            discounted_rewards = discount_reward(reward, discounted_r, cumul_r)
            policy, values = self.ppo.predictor.predict(
                {'input_states': state, 'context_input': context, 'empty': np.zeros((len(state), self.hidden_size))}, batch_size=256)
            advantages = np.zeros((episode_length, self.action_size))
            index = np.arange(episode_length)
            advantages[index, action] = 1
            A = discounted_rewards - values[:, 0]

            if len(total_state) == 0:

                total_state = state
                if context.shape[1] == memory.max_agents:
                    total_context = context
                else:
                    total_context = tf.keras.preprocessing.sequence.pad_sequences(
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
                    total_context = np.append(total_context, tf.keras.preprocessing.sequence.pad_sequences(
                        context, memory.max_agents, dtype='float32'), axis=0)
                total_reward = np.append(
                    total_reward, discounted_rewards, axis=0)
                total_A = np.append(total_A, A, axis=0)
                total_advantage = np.append(
                    total_advantage, advantages, axis=0)
                total_policy = np.append(total_policy, policy, axis=0)

        total_A = (total_A - total_A.mean())/(total_A.std() + 1e-8)
        self.model.fit({'input_states': total_state, 'context_input': total_context, 'empty': np.zeros((total_length, self.hidden_size)), 'A': total_A, 'old_pred': total_policy}, {
                       'policy_out': total_advantage, 'value_out': total_reward}, shuffle=True, batch_size=total_state.shape[0], epochs=8, verbose=0)

        memory.max_agents = 0
        memory.experience = {}

    def save(self, path="default"):
        PATH = "models/"+path

        print("Attempting to save model to: {}".format(PATH))
        self.model.save_weights(PATH)
        print("Success: Model Saved!")

    def load(self, path="default"):
        PATH = "models/"+path
        try:
            print("Attempting to load model from: {}".format(PATH))
            self.model.load_weights(PATH)
            print("Success: Model Loaded!")
        except Exception:
            print("There was an error loading the model from: {}".format(PATH))
            raise Exception
