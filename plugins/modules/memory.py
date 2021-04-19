import numpy as np
from tensorflow import keras
from bluesky import traf


class Memory():
    def __init__(self):
        self.max_agents = 0
        self.experience = {}

    def store(self, _id, state, action, nearest_ac, T=0):
        reward, done = self.get_reward(T, action, nearest_ac)

        state, context = state

        state = state.reshape((1, 6))
        context = context.reshape((1, -1, 6))

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

    def get_reward(self, T, action, nearest_ac):
        end = True
        if T == 1:
            reward = - 5
        elif T == 2:
            reward = 5
        else:
            end = False
            reward = 0

            if float(nearest_ac[0]) < 40 and float(nearest_ac[1]) <= 2000:
                dist = max(float(nearest_ac[0]), 10)
                alt = float(nearest_ac[1])

                dist_factor = -(1/dist)*10
                alt_factor = (0.1**(alt/2000))
                # alt_factor = 1

                # print(dist_factor, alt_factor)
                reward = dist_factor

            # if not (action == 1 or action == 3):
            #     reward += -0.15
                # reward += -0.75

        print(T, action, reward)
        return reward, end
