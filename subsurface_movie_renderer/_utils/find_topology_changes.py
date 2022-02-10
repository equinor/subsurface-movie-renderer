import numpy as np

from .distance_to_edge import distance_to_edge


def find_topology_changes(S1, S2, dx, dy):

    [M, N] = np.shape(S1)

    S12 = np.zeros((M, N))
    S12[S1 + S2 == 1] = 1

    # FIND "ISLANDS":

    reached = np.zeros((M, N))
    queue = []
    for i in range(M)[1:-1]:
        for j in range(N)[1:-1]:
            if S1[i, j] == 1:
                queue.append((i, j))

    while len(queue) > 0:
        new_cell = queue.pop()
        i = new_cell[0]
        j = new_cell[1]

        reached[i, j] = 1
        for delta_i in [-1, 0, 1]:
            for delta_j in [-1, 0, 1]:
                if (
                    (i + delta_i) >= 0
                    and (i + delta_i) < M
                    and (j + delta_j) >= 0
                    and (j + delta_j) < N
                ):
                    if (
                        S2[i + delta_i, j + delta_j] == 1
                        and reached[i + delta_i, j + delta_j] == 0
                    ):
                        queue.append((i + delta_i, j + delta_j))

    not_reached_islands = np.copy(S2)
    not_reached_islands[reached == 1] = 0

    # FIND "HOLES":

    S1 = 1 - np.copy(S1)
    S2 = 1 - np.copy(S2)

    reached = np.zeros((M, N))
    queue = []
    for i in range(M)[1:-1]:
        for j in range(N)[1:-1]:
            if S1[i, j] == 1:
                queue.append((i, j))

    while len(queue) > 0:
        new_cell = queue.pop()
        i = new_cell[0]
        j = new_cell[1]

        reached[i, j] = 1
        for delta_i in [-1, 0, 1]:
            for delta_j in [-1, 0, 1]:
                if (
                    (i + delta_i) >= 0
                    and (i + delta_i) < M
                    and (j + delta_j) >= 0
                    and (j + delta_j) < N
                ):
                    if (
                        S2[i + delta_i, j + delta_j] == 1
                        and reached[i + delta_i, j + delta_j] == 0
                    ):
                        queue.append((i + delta_i, j + delta_j))

    not_reached_holes = np.copy(S2)
    not_reached_holes[reached == 1] = 0

    # CHOOSE STARTING POINTS:

    not_reached = not_reached_islands + not_reached_holes
    distance = distance_to_edge(not_reached, dx, dy)  # calculate distance map to edge

    new_indices = []
    while np.max(np.max(distance)) > 0:
        index = np.unravel_index(np.argmax(distance), (M, N))
        new_indices.append(index)

        queue = [index]
        while len(queue) > 0:
            index = queue.pop()
            i = index[0]
            j = index[1]
            distance[i, j] = 0
            for delta_i in [-1, 0, 1]:
                for delta_j in [-1, 0, 1]:
                    if distance[i + delta_i, j + delta_j] > 0:
                        queue.append((i + delta_i, j + delta_j))

    return new_indices
