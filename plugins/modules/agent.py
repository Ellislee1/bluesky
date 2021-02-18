import numpy as np
from bluesky.tools.aero import ft
from bluesky.tools.geo import latlondist, nm
from tensorflow import keras
from modules.ppo import PPO


HIDDEN_SIZE = 32


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
            alt = traf.alt[new_local[i]]/ft
            tas = traf.tas[new_local[i]]
            vs = traf.vs[new_local[i]]
            trk = traf.trk[new_local[i]]

            closest.append([lat, lon, alt, tas, vs, trk])
    closest = closest
    for i in range(5-len(closest)):
        closest.append([0, 0, 0, 0, 0, 0])

    return closest


# Get the nearest aircraft to the agent
def nearest_ac(dist_matrix, _id, traf):
    idx = traf.id2idx(_id)

    closest = dist_matrix[0]
    close = get_dist([traf.lat[idx], traf.lon[idx]], [
                     closest[0], closest[1]])
    this_alt = traf.alt[idx]/ft
    close_alt = closest[2]
    alt_separations = abs(this_alt - close_alt)

    if close == 0 and alt_separations == this_alt:
        return (-1, -1)
    else:
        return close, alt_separations


class Agent:
    def __init__(self, statesize, actionsize, valuesize, num_intruders=5):
        self.num_intruders = num_intruders

        self.model = PPO(statesize, num_intruders, actionsize, valuesize)

    def terminal(self, traf, i, nearest, traffic, memory):
        # get ac index in traffic array
        _id = traf.id[i]
        """
            0 = not terminal
            1 = collision
            2 = goal reached
        """
        if nearest:
            dist, alt = nearest
        else:
            dist, alt = 10e+11, 10e+11

        d_goal = get_goal_dist(_id, traf, traffic)
        memory.dist_goal[_id] = d_goal

        memory.dist_close[_id] = dist
        # Linear goal distance
        g_dist = get_dist([traf.lat[i], traf.lon[i]],
                          traffic.routes[_id][-1])
        # T = Terminal type
        T = self.is_terminal(dist, alt, g_dist)

        return T

    def is_terminal(self, distance, v_sep, d_goal):
        if distance <= 5 and v_sep < 2000:
            return 1

        if d_goal <= 15:
            return 2

        return 0

    def act(self, state, context):
        context = context.reshape((state.shape[0], -1, 7))

        if context.shape[1] > self.num_intruders:
            context = context[:, -self.num_intruders:, :]
        elif context.shape[1] < self.num_intruders:
            context = keras.preprocessing.sequence.pad_sequences(
                context, self.num_intruders, dtype='float32')

        policy, value = self.model.estimator.predict({'input_state': state, 'input_context': context, 'empty': np.zeros(
            (state.shape[0], HIDDEN_SIZE))}, batch_size=state.shape[0])

        return policy, value
