from modules.ppo import PPO


class Agent:
    def __init__(self, statesize, num_intruders, actionsize, valuesize):
        self.num_intruders = num_intruders

        self.model = PPO(statesize, num_intruders, actionsize, valuesize)
