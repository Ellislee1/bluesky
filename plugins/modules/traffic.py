import numpy as np
import random
# Import the global bluesky objects. Uncomment the ones you need
from bluesky.stack import stack
from bluesky.tools import areafilter


class Traffic():
    def __init__(self, max_ac=500, times=[20, 25, 30], iata="SWN", types=['A320'], max_spd=320, min_spd=250, max_alt=32000, min_alt=28000, network=None):

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
        self.routing = {}

        self.first_run = True

        self.spawn_queue = random.choices(
            self.times, k=len(self.network.airports))

        print("Traffic: READY")

    def spawn(self):
        if self.first_run == True:
            self.first()
        elif self.total < self.max_ac:
            for k in range(len(self.spawn_queue)):
                if self.update_timer == self.spawn_queue[k]:
                    origin = [*self.network.airports.keys()][k]

                    path, route_coords = self.network.generate_route(origin)

                    self.create_ac(route_coords, path)
                    self.spawn_queue[k] = self.update_timer + \
                        random.choices(self.times, k=1)[0]

                if self.total >= self.max_ac:
                    break

        self.update_timer += 1

    def first(self):
        for i, origin in enumerate(self.network.airports):
            if self.total < self.max_ac:
                path, route_coords = self.network.generate_route(origin)
                self.create_ac(route_coords, path)
        self.first_run = False

    def create_ac(self, route_coords, path):
        callsign = self.iata + str(self.total)
        self.routes.update({callsign: route_coords[:]})
        self.routing.update({callsign: path[:]})

        node = route_coords.pop(0)
        ac_type = random.choice(self.types)
        s_lat, s_lon = node[0], node[1]
        hdg = self.network.get_heading(node, route_coords[0])
        alt = np.random.randint(self.min_alt, self.max_alt)
        spd = np.random.randint(self.min_spd, self.max_spd)

        stack("CRE {} {} {},{} {} {} {}".format(
            callsign, ac_type, s_lat, s_lon, hdg, alt, spd))

        while route_coords:
            node = route_coords.pop(0)

            stack("ADDWPT {} {},{}".format(callsign, node[0], node[1]))

        self.active += 1
        self.total += 1

    def check_done(self):
        if self.total >= self.max_ac and self.active == 0:
            return True
        return False

    def get_sectors(self, sectors, traf):
        self.traf_in_sectors = []

        for sector in sectors:
            self.traf_in_sectors.append(areafilter.checkInside(
                sector, traf.lat, traf.lon, traf.alt))

        self.traf_in_sectors = np.array(self.traf_in_sectors)

    def get_in_sectors(self, _id, traf):
        idx = traf.id2idx(_id)

        what_sectors = []
        for i, sector in enumerate(self.traf_in_sectors):
            if sector[idx] == True:
                what_sectors.append(i)

        return what_sectors

    def get_active(self, traf):
        active = []

        for i in range(len(traf.id)):
            temp = []
            for x, sector in enumerate(self.traf_in_sectors):
                if sector[i] == True:
                    temp.append(x)
            active.append(np.array(temp))

        return np.array(active)

    def get_sector_indexes(self, sector_idx, _id, traf):
        idx = traf.id2idx(_id)
        watch_ac = []

        for i, val in enumerate(self.traf_in_sectors[sector_idx]):
            if val and not i == idx:
                watch_ac.append(i)

        return(watch_ac)

    def reset(self):
        self.active = 0
        self.total = 0
        self.update_timer = 0
        self.routes = {}

        self.first_run = True

        self.spawn_queue = random.choices(
            self.times, k=len(self.network.airports))
