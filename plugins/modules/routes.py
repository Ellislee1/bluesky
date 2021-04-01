from bluesky.tools.geo import latlondist, nm
from bluesky.stack import stack
from geographiclib.geodesic import Geodesic
import json
import random
import numpy as np


class Route_Manager():
    def __init__(self, PATH, mindist=30, test_routes=False, draw_paths=False):
        # Min distance the ac must travel before reaching an terminal node
        self.min_dist = mindist

        self.initilise(PATH)

        if test_routes:
            self.test_paths()

        if draw_paths:
            self.draw_routes()

        print("Airspace: READY")

    # Initilise the system
    def initilise(self, path):
        with open(path) as PATH:
            all_nodes = json.load(PATH)["nodes"]

        # Load all departure, arrival and en route nodes fromt he JSON file
        all_dep = all_nodes["dep"]
        all_arr = all_nodes["arr"]
        routes = all_nodes["routes"]

        self.nodes = {}
        self.departure = {}
        self.arrival = {}

        # Add nodes to the dictionaries
        for dep in all_dep:
            self.departure[dep["id"]] = dep

        for arr in all_arr:
            self.arrival[arr["id"]] = arr

        for node in routes:
            self.nodes[node["id"]] = node

        # Create an indexed array of all nodes (Used for passing routes to the nn)
        self.idx_array = np.concatenate(
            (np.array(list(self.departure.keys())), np.array(list(self.arrival.keys())), np.array(
                list(self.nodes.keys()))))

        print(self.idx_array)

    # visualise the routes
    def draw_routes(self):
        for x in self.departure:
            ap = self.departure[x]
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

        for x in self.arrival:
            nd = self.arrival[x]
            for connection in nd["connection"]:
                try:
                    node = self.nodes[connection]
                    stack(
                        f'LINE {nd["id"]}{node["id"]} {nd["lat"]},{nd["lon"]} {node["lat"]},{node["lon"]}')
                except:
                    print(f"No node: {connection}")

    # Test that path are valid
    def test_paths(self):
        for origin in (self.departure):
            for terminal in (self.arrival):
                path = self.find_path(origin, terminal)
                if len(path) <= 1:
                    print(f'{origin}-{terminal}: NO Path')
                else:
                    print(f'{origin}-{terminal}: {path}')

    # Find a path between two points
    def find_path(self, origin, terminal):
        routes = self.bfs(origin)
        path = self.get_path(routes, terminal)
        return path

    # BFS to find routing path
    def bfs(self, origin):
        visited = []
        queue = []
        routes = {}

        visited.append(origin)

        # Check that the connection has not been visited and append it
        for connection in self.departure[origin]["connection"]:
            if connection not in visited and connection not in self.departure:
                visited.append(connection)
                routes[connection] = origin
                queue.append(connection)

        # Loop through the queue adding all routes
        while queue:
            node = queue.pop(0)
            for connection in self.nodes[node]["connection"]:

                if connection not in visited and connection not in self.departure and connection not in self.arrival:
                    visited.append(connection)
                    routes[connection] = node
                    queue.append(connection)
                if connection not in visited and (connection in self.departure or connection in self.arrival):
                    visited.append(connection)
                    routes[connection] = node
        return routes

    # Generate a ptath
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

    # Generate a route
    def generate_route(self, dep):
        dist = 0
        while dist <= self.min_dist:
            arr = random.choice(list(self.arrival.values()))["id"]
            dist = self.get_dist(dep, arr)

        path = self.find_path(dep, arr)
        return path

    # Get the coordinates of a waypoint
    def get_coords(self, _id):
        if _id in self.departure.keys():
            return [self.departure[_id]["lat"], self.departure[_id]["lon"]]

        if _id in self.arrival.keys():
            return [self.arrival[_id]["lat"], self.arrival[_id]["lon"]]

        if _id in self.nodes.keys():
            return [self.nodes[_id]["lat"], self.nodes[_id]["lon"]]

        raise Exception(
            "Unable to finde coordinates for {}. The waypoint does not exist as a node, departure point or arrival point.".format(_id))

    # Get the lateral (shortest) distance between two waypoints
    def get_dist(self, a, b):
        a_port = self.departure[a]
        b_port = self.arrival[b]
        return latlondist(a_port["lat"], a_port["lon"], b_port["lat"], b_port["lon"])/nm

    # Get the heading between two way points
    def get_heading(self, start, _next):
        x, y = start[0], start[1]
        a, b = _next[0], _next[1]
        return Geodesic.WGS84.Inverse(x, y, a, b)['azi1']
