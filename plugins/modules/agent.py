import numpy as np
from bluesky.tools.aero import ft
from bluesky.tools.geo import latlondist, nm

from modules.ppo import PPO


# Get the distance between two points.
def get_dist(pos1, pos2):
    return latlondist(pos1[0], pos1[1], pos2[0], pos2[1])/nm


# Get the route length to the goal
def get_goal_dist(_id, traf, traffic):
    idx = traf.id2idx(_id)
    active_waypoint = [traf.actwp.lat[idx], traf.actwp.lon[idx]]
    route = traffic.routes[_id]

    start = 0
    found = False
    min_dist = 9999
    i = 1
    while not found:
        dist = get_dist(active_waypoint, route[i])
        if dist > min_dist:
            found = True
        elif dist < min_dist:
            min_dist = dist
            start = i
        i += 1
        if i >= len(route):
            found = True

    d_goal = 0
    current = [traf.lat[idx], traf.lon[idx]]
    for i in range(start, len(route)):
        d_goal += get_dist(current, route[i])
        current = route[i]
    return d_goal

# Get a distance matrix


def get_distance_matrix_ac(traf, _id, local_traf):
    idx = traf.id2idx(_id)
    distances = []

    for ac in local_traf:
        distances.append(latlondist(
            traf.lat[idx], traf.lon[idx], traf.lat[ac], traf.lon[ac])/nm)

    return distances


# Get the nearest n aircraft
def get_nearest_n(dist_matrix, _id, traf, local_traf, n=4):
    closest = []
    new_dist_traf = []
    new_local_traf = []
    if len(dist_matrix) > 0:
        for i, dist in enumerate(dist_matrix):
            if dist <= 100:
                new_dist_traf.append(dist)
                new_local_traf.append(local_traf[i])

    new_dist = new_dist_traf.copy()
    new_dist.sort()
    new_local = []
    for i in new_dist:
        new_local.append(new_local_traf[new_dist_traf.index(i)])

    if new_dist:
        for i in range(0, min(len(new_dist), n)):
            lat = traf.lat[new_local[i]]
            lon = traf.lon[new_local[i]]

            closest.append([lat, lon, traf.alt[new_local[i]]/ft, new_dist[i]])
    closest = closest
    for i in range(4-len(closest)):
        closest.append([0, 0, 0, 0])

    return closest


# Get the nearest aircraft to the agent
def nearest_ac(dist_matrix, _id, traf):
    idx = traf.id2idx(_id)

    closest = dist_matrix[0]
    close = closest[3]
    this_alt = traf.alt[idx]/ft
    close_alt = closest[2]
    alt_separations = abs(this_alt - close_alt)

    if close == 0 and alt_separations == this_alt:
        print(close, "HERE")
        return (-1, -1)
    else:
        return close, alt_separations


class Agent:
    def __init__(self, statesize, num_intruders, actionsize, valuesize):
        self.num_intruders = num_intruders

        self.model = PPO(statesize, num_intruders, actionsize, valuesize)

    def terminal(self, traf, _id, local_traf, traffic, memory):
        # get ac index in traffic array
        idx = traf.id2idx(_id)
        """
            0 = not terminal
            1 = collision
            2 = goal reached
        """

        if len(local_traf) > 0:
            dist_matrix = get_distance_matrix_ac(traf, _id, local_traf)

            nearest_n = get_nearest_n(get_distance_matrix_ac(
                traf, _id, local_traf), _id, traf, local_traf)

            distance, v_separation = nearest_ac(nearest_n, _id, traf)
        else:
            distance, v_separation = -1, -1

        memory.dist_close[_id] = distance

        d_goal = get_goal_dist(_id, traf, traffic)
        memory.dist_goal[_id] = d_goal

        # Linear goal distance
        g_dist = get_dist([traf.lat[idx], traf.lon[idx]],
                          traffic.routes[_id][-1])
        # T = Terminal type
        if len(local_traf) <= 0:
            if g_dist <= 15:
                T = 2
            else:
                T = 0
        else:
            T = self.is_terminal(distance, v_separation, g_dist)

        return T

    def is_terminal(self, distance, v_sep, d_goal):
        if distance <= 5 and v_sep < 2000:
            return 1

        if d_goal <= 15:
            return 2

        return 0

    def get_reward(self):
        pass

    def store(self):
        pass
