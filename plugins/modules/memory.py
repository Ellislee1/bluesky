import numpy as np
from tensorflow import keras


class Memory:
    def __init__(self):
        self.dist_goal = {}
        self.dist_close = {}
        self.previous_observation = {}
        self.previous_action = {}
        self.experience = {}
        self.observation = {}

        self.max_agents = 0

    def clear_memory(self):
        self.dist_goal = {}
        self.dist_close = {}
        self.previous_observation = {}
        self.previous_action = {}
        # self.experience = {}
        self.observation = {}

    def store(self, state, action, next_state, traf, _id, nearest_ac, T=0):
        reward = 0
        done = False

        idx = traf.id2idx(_id)

        dist, alt = nearest_ac

        dist_flown = traf.distflown[idx]

        if T == 0:

            if (dist < 10 and alt < 2500):
                reward -= (1-(alt/2500))**(1-(dist/10))

        else:
            done = True

            if T == 1:
                reward -= 5
            elif T == 2:
                reward = 5

        if T == 0 and reward < 0:
            reward -= 1

        state, context = state

        state = state.reshape((1, 9))
        context = context.reshape((1, -1, 10))

        if context.shape[1] > 5:
            context = context[:, -5:, :]

        self.max_agents = max(self.max_agents, context.shape[1])

        if not _id in self.experience.keys():
            self.experience[_id] = {}

        try:
            self.experience[_id]['state'] = np.append(
                self.experience[_id]['state'], state, axis=0)

            if self.max_agents > self.experience[_id]['context'].shape[1]:
                self.experience[_id]['context'] = np.append(keras.preprocessing.sequence.pad_sequences(
                    self.experience[_id]['context'], self.max_agents, dtype='float32'), context, axis=0)
            else:
                self.experience[_id]['context'] = np.append(self.experience[_id]['context'], keras.preprocessing.sequence.pad_sequences(
                    context, self.max_agents, dtype='float32'), axis=0)

            self.experience[_id]['action'] = np.append(
                self.experience[_id]['action'], action)
            self.experience[_id]['reward'] = np.append(
                self.experience[_id]['reward'], reward)
            self.experience[_id]['done'] = np.append(
                self.experience[_id]['done'], done)
        except:
            self.experience[_id]['state'] = state
            if self.max_agents > context.shape[1]:
                self.experience[_id]['context'] = keras.preprocessing.sequence.pad_sequences(
                    context, self.max_agents, dtype='float32')
            else:
                self.experience[_id]['context'] = context

            self.experience[_id]['action'] = [action]
            self.experience[_id]['reward'] = [reward]
            self.experience[_id]['done'] = [done]

        return reward
