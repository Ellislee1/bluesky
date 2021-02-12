""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
import datetime
import time
from random import randint

# Import the global bluesky objects. Uncomment the ones you need
import bluesky
import numpy as np
from bluesky import core, stack, traf  # , settings, navdb, sim, scr, tools
from bluesky.tools import areafilter
from bluesky.tools.aero import ft, kts
from bluesky.tools.geo import latlondist, nm

from modules.agent import Agent
from modules.airspace import Airspace
from modules.memory import Memory
from modules.sectors import load_sectors
from modules.traffic import Traffic

EPOCHS = 2500
MAX_AC = 80
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
                           5, 5)

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
        if not self.initilized:
            self.init()

        if not self.start:
            self.start = time.perf_counter()

        self.traffic.spawn()

        self.traffic.get_sectors(self.sectors, traf)

        if len(traf.id) > 0:
            terminal, actions, reward = self.agent.update(
                traf, self.traffic, self.memory)
            self.handle_actions(actions)
            terminal = self.traffic.handle_terminal(terminal)
            self.mean_rewards.append(reward)

            if len(terminal) > 0:
                self.success += sum(1 for i in terminal if i == 1)
                self.fail += sum(1 for i in terminal if i == 2)

        if self.traffic.check_done():
            self.epoch_reset()

    def handle_actions(self, actions):
        for i, action in enumerate(actions):
            if action == 1:
                stack.stack('{} ALT {}'.format(
                    traf.id[i], self.get_new_alt(i, self.actions[1])))
            elif action == 3:
                stack.stack('{} SPD {}'.format(
                    traf.id[i], self.get_new_spd(i, self.actions[3])))
            elif action == 0:
                pass
            elif action == 4:
                stack.stack('{} SPD {}'.format(
                    traf.id[i], self.get_new_spd(i, self.actions[4])))
            elif action == 2:
                stack.stack('{} ALT {}'.format(
                    traf.id[i], self.get_new_alt(i, self.actions[2])))
            else:
                print(traf.id[i], "Error")

    def get_new_alt(self, _idx, change_val):
        return np.clip((traf.alt[_idx]/ft)+change_val, a_min=22000, a_max=36000)

    def get_new_spd(self, _idx, change_val):
        return np.clip((traf.cas[_idx]/kts)+change_val, a_min=300, a_max=380)

    def epoch_reset(self):

        self.traffic.reset()
        self.stop = time.perf_counter()

        self.all_success.append(self.success)
        self.all_fail.append(self.fail)

        self.all_mean_success = np.mean(self.all_success)

        if (self.epoch_counter+1) >= 50:
            self.mean_success = np.mean(self.all_success[-50:])

        if (self.epoch_counter+1) % 5 == 0:
            if self.mean_success >= self.best:
                print('----- Saving New Best Model -----')
                self.agent.save(PATH)

            print('----- Training Model -----')
            self.agent.update_PPO(self.memory)
            self.memory.clear_memory()

        self.best = max(self.mean_success, self.best)

        string = "Epoch run in {:.2f} seconds".format(self.stop-self.start)
        self.print_all(string)
        string = "Success: {} | Fail: {} | Mean Success: {:.4f} | Mean Reward {:.2f} | (50) Mean Success Rolling {:.4f} | Best {:.4f}".format(
            self.success, self.fail, self.all_mean_success, np.mean(self.mean_rewards), self.mean_success, self.best)
        self.print_all(string)

        if self.epoch_counter+1 >= EPOCHS:
            stack.stack("STOP")

        self.epoch_counter += 1
        string = "=================================\n   UPDATE: RUNNING EPOCH {}\n=================================\n".format(
            self.format_epoch())
        self.print_all(string)

        self.success = 0
        self.fail = 0
        self.stop = None
        self.start = None
        self.mean_rewards = []

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
