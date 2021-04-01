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
        self.routes = np.zeros(self.max_ac)

        self.first_run = True

        self.spawn_queue = random.choices(
            self.times, k=len(self.network.all_routes))

        print("Traffic: READY")

    # Creates aircraft at each node for the start
    def first(self):
        for i in range(len(self.network.all_routes)):
            if self.total < self.max_ac:
                path = self.network.all_routes[i]
                self.create_ac(path, i)
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
                    path = self.network.all_routes[k]
                    # Create the ac and reset the timer for this ac
                    self.create_ac(path, k)
                    self.spawn_queue[k] = self.update_timer + \
                        random.choices(self.times, k=1)[0]

                # Do not let it go over the max
                if self.total >= self.max_ac:
                    break

        self.update_timer += 1

    # Creating an aircraft in the system
    def create_ac(self, path, route_i):
        callsign = self.iata + str(self.total)
        self.routes[self.total] = route_i

        ac_type = random.choice(self.types)

        path = self.network.all_routes[route_i]["points"].copy()

        node = path.pop(0)

        hdg = self.network.all_routes[route_i]["heading"]
        alt = np.random.randint(self.min_alt, self.max_alt)
        spd = np.random.randint(self.min_spd, self.max_spd)

        stack("CRE {} {} {},{} {} {} {}".format(
            callsign, ac_type, node[0], node[1], hdg, alt, spd))

        while path:
            node = path.pop(0)
            stack("ADDWPT {} {},{}".format(
                callsign, node[0], node[1]))

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
        self.routes = np.zeros(self.max_ac)

        self.first_run = True

        self.spawn_queue = random.choices(
            self.times, k=len(self.network.all_routes))
