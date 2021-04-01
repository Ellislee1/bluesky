import random

import numpy as np
from bluesky.stack import stack
from bluesky.tools import areafilter
from bluesky import traf


class Traffic_Manager():
    def __init__(self, max_ac=20, times=[20, 25, 30], iata="SWN", types=["A320"], max_spd=320,
                 min_spd=250, max_alt=32000, min_alt=28000, network=None):
        self.max_ac = max_ac
        self.times = times
        self.iata = iata
        self.types = types
        self.max_spd = max_spd+1
        self.min_spd = min_spd
        self.max_alt = max_alt+1
        self.min_alt = min_alt
        self.network = network

        self.active = 0
        self.total = 0
        self.update_timer = 0
        self.routes = {}

        self.first_run = True

        self.spawn_queue = random.choices(
            self.times, k=len(self.network.departure))

        print("Traffic: READY")

    # Creates aircraft at each node for the start
    def first(self):
        for i, origin in enumerate(self.network.departure):
            if self.total < self.max_ac:
                path = self.network.generate_route(origin)
                self.create_ac(path)
        self.first_run = False

    # Run on update, create new aircraft
    def spawn(self):
        if self.first_run == True:
            # First Run
            self.first()

        elif self.total < self.max_ac:
            # Run if the total number of aircraft for the epoch has npt been generated
            for k in range(len(self.spawn_queue)):
                if self.update_timer == self.spawn_queue[k]:
                    # Get the origin from the queue
                    origin = [*self.network.departure.keys()][k]

                    # Generate a path
                    path = self.network.generate_route(origin)

                    # Create the ac and reset the timer for this ac
                    self.create_ac(path)
                    self.spawn_queue[k] = self.update_timer + \
                        random.choices(self.times, k=1)[0]

                # Do not let it go over the max
                if self.total >= self.max_ac:
                    break

        self.update_timer += 1

    # Creating an aircraft in the system
    def create_ac(self, path):
        callsign = self.iata + str(self.total)
        self.routes.update({callsign: path[:]})

        node = path.pop(0)
        node_coord = self.network.get_coords(node)
        ac_type = random.choice(self.types)
        s_lat, s_lon = node_coord[0], node_coord[1]
        hdg = self.network.get_heading(
            node_coord, self.network.get_coords(path[0]))
        alt = np.random.randint(self.min_alt, self.max_alt)
        spd = np.random.randint(self.min_spd, self.max_spd)

        stack("CRE {} {} {},{} {} {} {}".format(
            callsign, ac_type, s_lat, s_lon, hdg, alt, spd))

        while path:
            node = path.pop(0)
            node_coord = self.network.get_coords(node)
            stack("ADDWPT {} {},{}".format(
                callsign, node_coord[0], node_coord[1]))

        self.active += 1
        self.total += 1

    # Get the sector information for all aircraft
    def get_sectors(self, sectors):
        self.traf_in_sectors = []

        for sector in sectors:
            self.traf_in_sectors.append(areafilter.checkInside(
                sector, traf.lat, traf.lon, traf.alt))

        self.traf_in_sectors = np.array(self.traf_in_sectors)

    # get the sectors each aircraft is in
    def update_active(self, sectors):
        # Get full array of each sector and aircraft in that sector
        self.get_sectors(sectors)
        active = []

        # See if the aircraft is in that sector
        for i in range(len(traf.id)):
            temp = []
            for x, sector in enumerate(self.traf_in_sectors):
                if sector[i] == True:
                    temp.append(x)
            active.append(np.array(temp))

        self.active_sectors = np.array(active)

    # Check to see if the end conditions (max aircraft has been reached and scenario is empty)
    def check_done(self):
        if self.total >= self.max_ac and self.active == 0:
            return True
        return False

    # Reset the traffic manager for the next epoch
    def reset(self):
        self.active = 0
        self.total = 0
        self.update_timer = 0
        self.routes = {}

        self.first_run = True

        self.spawn_queue = random.choices(
            self.times, k=len(self.network.departure))
