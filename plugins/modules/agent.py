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
    active_waypoint = [traf.actwp.lat[idx], traf.actwp.lat[idx]]
    route = traffic.routes[_id]

    start = 0
    found = False
    min_dist = 9999
    i = -1
    while not found:
        dist = get_dist(active_waypoint, route[i])
        if not dist < min_dist:
            found = True
        else:
            min_dist = dist
            i += 1

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


# Get the nearest aircraft to the agent
def nearest_ac(dist_matrix, _id, traf):
    idx = traf.id2idx(_id)

    closest = dist_matrix[0]
    close = closest[6]
    this_alt = traf.alt[idx]/ft
    close_alt = closest[1]
    alt_separations = abs(this_alt - close_alt)

    if close == 0 and alt_separations == this_alt:
        return (-1, -1)
    else:
        return close, alt_separations


class Agent:
    def __init__(self, statesize, num_intruders, actionsize, valuesize):
        self.num_intruders = num_intruders

        self.model = PPO(statesize, num_intruders, actionsize, valuesize)

    def update(self, traf, _id, local_traf, traffic):
        # get ac index in traffic array
        idx = traf.id2idx(_id)
        """
            0 = not terminal
            1 = collision
            2 = goal reached
        """
        dist_matrix = get_distance_matrix_ac(traf, _id, local_traf)

        distance, v_separation = nearest_ac(dist_matrix, _id, traf)

        d_goal = get_goal_dist(_id, traf, traffic)
        print(_id, d_goal)
        # T = Terminal type
        T = is_terminal(distance, v_separation, d_goal)

    def is_terminal(self, distance, v_sep, d_goal):
        if d_goal < 12:
            return 2

        if distance <= 5 and v_sep < 2000:
            return 1

        return 0
