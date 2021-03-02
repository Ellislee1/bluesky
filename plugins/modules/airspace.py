from bluesky.tools.geo import latlondist, nm
from bluesky.stack import stack
from geographiclib.geodesic import Geodesic
import json
import random
import numpy as np


class Airspace():
    def __init__(self, path="nodes/nodes.json", min_dist=30):
        self.min_dist = min_dist
        self.initilise(path)
        # self.draw_routes()
        # self.test_paths()
        self.array = np.concatenate(
            (np.array(list(self.airports.keys())), np.array(
                list(self.nodes.keys()))))

    def initilise(self, path):
        with open(path) as PATH:
            all_nodes = json.load(PATH)["nodes"]

        all_airports = all_nodes["airports"]
        routes = all_nodes["routes"]

        self.nodes = {}
        self.airports = {}

        for airport in all_airports:
            self.airports[airport["id"]] = airport

        for node in routes:
            self.nodes[node["id"]] = node

        print("Airspace: READY")

    def generate_route(self, dep):
        dist = 0
        while dist <= self.min_dist:
            arr = random.choice(list(self.airports.values()))["id"]
            dist = self.get_dist(dep, arr)

        path = self.find_path(dep, arr)
        route_coords = self.get_coords(path)
        return path, route_coords

    def get_dist(self, a, b):
        a_port = self.airports[a]
        b_port = self.airports[b]
        return latlondist(a_port["lat"], a_port["lon"], b_port["lat"], b_port["lon"])/nm

    def draw_routes(self):
        for x in self.airports:
            ap = self.airports[x]
            for connection in ap["connection"]:
                node = self.nodes[connection]
                stack(
                    f'LINE {ap["id"]}{node["id"]} {ap["lat"]},{ap["lon"]} {node["lat"]},{node["lon"]}')

        for x in self.nodes:
            nd = self.nodes[x]
            for connection in nd["connection"]:
                try:
                    node = self.nodes[connection]
                    stack(
                        f'LINE {nd["id"]}{node["id"]} {nd["lat"]},{nd["lon"]} {node["lat"]},{node["lon"]}')
                except:
                    print(f"No node: {connection}")

    def test_paths(self):
        for origin in (self.airports):
            for terminal in (self.airports):
                path = self.find_path(origin, terminal)
                if len(path) <= 1:
                    print(f'{origin}-{terminal}: NO Path')
                else:
                    print(f'{origin}-{terminal}: {path}')

    def find_path(self, origin, terminal):
        routes = self.bfs(origin)
        path = self.get_path(routes, terminal)
        return path

    def bfs(self, origin):
        visited = []
        queue = []
        routes = {}

        visited.append(origin)

        for connection in self.airports[origin]["connection"]:
            if connection not in visited and connection not in self.airports:
                visited.append(connection)
                routes[connection] = origin
                queue.append(connection)

        while queue:
            node = queue.pop(0)
            for connection in self.nodes[node]["connection"]:
                if connection not in visited and connection not in self.airports:
                    visited.append(connection)
                    routes[connection] = node
                    queue.append(connection)
                if connection not in visited and connection in self.airports:
                    visited.append(connection)
                    routes[connection] = node
        return routes

    def get_path(self, routes, terminal):
        path = []
        end = False
        while not end:
            try:
                path.insert(0, terminal)
                terminal = routes[terminal]
            except:
                end = True
        return path

    def get_heading(self, start, _next):
        x, y = start[0], start[1]
        a, b = _next[0], _next[1]
        return Geodesic.WGS84.Inverse(x, y, a, b)['azi1']

    def get_coords(self, route):
        route_coords = []

        for pos in route:
            try:
                route_coords.append([
                    self.airports[pos]["lat"],
                    self.airports[pos]["lon"]]
                )
            except:
                route_coords.append(
                    [self.nodes[pos]["lat"], self.nodes[pos]["lon"]])
        return route_coords
