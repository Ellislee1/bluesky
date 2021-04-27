import json

import numpy as np
from bluesky import traf
from bluesky.stack import stack
from bluesky.tools import areafilter


class Sector_Manager():
    def __init__(self, path):
        self.load_sectors(path)

    def load_sectors(self, path):
        print("Initilizing Sectors")
        with open(path) as PATH:
            sectors = json.load(PATH)["sectors"]

        self.system_sectors = {}
        for sector in sectors:
            self.system_sectors[sector["name"]] = Sector(sector)

        for sector in self.system_sectors:
            self.system_sectors[sector].initilise()

    def update_ac_sectors(self):
        self.traf_in_sectors = []

        for sector in self.system_sectors:
            self.traf_in_sectors.append(areafilter.checkInside(
                sector, traf.lat, traf.lon, traf.alt))

        self.traf_in_sectors = np.array(self.traf_in_sectors)

     # get the sectors each aircraft is in
    def update_active(self):
        # Get full array of each sector and aircraft in that sector
        self.update_ac_sectors()
        active = []

        # See if the aircraft is in that sector
        for i in range(len(traf.id)):
            temp = []
            for x, sector in enumerate(self.traf_in_sectors):
                if sector[i] == True:
                    temp.append(x)
            active.append(set(temp))

        self.active_sectors = np.array(active)


class Sector():
    def __init__(self, sector):
        self.name = sector["name"]
        self.points = sector["points"]
        self.constraints = sector["constraints"]

    def initilise(self):
        coords = None

        string = f'POLY {self.name}'
        origin = []
        for i, point in enumerate(self.points):
            if i == 0:
                origin = [point["lat"], point["lon"]]
                coords = []
            string += f' {point["lat"]},{point["lon"]}'
            coords = np.append(coords, [point["lat"], point["lon"]])
        coords = np.append(coords, [origin])
        # areafilter.defineArea(areaname=self.name,
        #                       areatype='POLY', coordinates=np.array(coords))
        stack(string)
