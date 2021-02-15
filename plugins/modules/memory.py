class Memory:
    def __init__(self):
        self.dist_goal = {}
        self.dist_close = {}

    def clear_memory(self):
        self.dist_goal = {}
        self.dist_close = {}
