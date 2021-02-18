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

    def clear_memory(self):
        self.dist_goal = {}
        self.dist_close = {}
        self.previous_observation = {}
        self.previous_action = {}
        # self.experience = {}
        self.observation = {}

    def store(self, state, action, next_state, traf, _id, nearest_ac, T=0, max_agents=0):
        reward = 0
        done = False

        idx = traf.id2idx(_id)

        dist, alt = nearest_ac

        if T == 0:
            if (dist > 5 and alt <= 2500) and (dist != -1):
                reward -= 0.5 * (alt/1000)
            elif dist <= 5 and alt <= 2500:
                reward -= 0.75 * (alt/1000)
        else:
            done = True

            if T == 1:
                reward -= 2

        state, context = state

        state.reshape((1, 6))
        context = context.reshape(1, -1, 7)

        if context.shape[1] > 5:
            context = context[:, -5:, :]

        max_agents = max(max_agents, context.shape[1])

        if not _id in self.experience.keys():
            self.experience[_id] = {}

        try:
            self.experience[_id]['state'] = np.append(
                self.experience[_id]['state'], state, axis=0)

            if max_agents > self.experience[_id]['context'].shape[1]:
                self.experience[_id]['context'] = np.append(keras.preprocessing.sequence.pad_sequences(
                    self.experience[_id]['context'], max_agents, dtype='float32'), context, axis=0)
            else:
                self.experience[_id]['context'] = np.append(self.experience[_id]['context'], keras.preprocessing.sequence.pad_sequences(
                    context, max_agents, dtype='float32'), axis=0)

            self.experience[_id]['action'] = np.append(
                self.experience[_id]['action'], action)
            self.experience[_id]['reward'] = np.append(
                self.experience[_id]['reward'], reward)
            self.experience[_id]['done'] = np.append(
                self.experience[_id]['done'], done)
        except:
            self.experience[_id]['state'] = state
            if max_agents > context.shape[1]:
                self.experience[_id]['context'] = keras.preprocessing.sequence.pad_sequences(
                    context, max_agents, dtype='float32')
            else:
                self.experience[_id]['context'] = context

            self.experience[_id]['action'] = [action]
            self.experience[_id]['reward'] = [reward]
            self.experience[_id]['done'] = [done]

        return max_agents
