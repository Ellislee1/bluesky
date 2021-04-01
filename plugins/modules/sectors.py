import json
import numpy as np

from bluesky.stack import stack
from bluesky.tools import areafilter


class Sector_Manager():
    def __init__(self, PATH):
        self.load_sectors(PATH)

    def load_sectors(self, path):
        print("Initilizing Sectors")
        with open(path) as PATH:
            sectors = json.load(PATH)["sectors"]

        self.system_sectors = {}
        for sector in sectors:
            self.system_sectors[sector["name"]] = Sector(sector)

        for sector in self.system_sectors:
            self.system_sectors[sector].initilise()


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
