import numpy as np
from operator import itemgetter

import networkx as nx


def directed_path_exists(DAG, startnode, endnode):
    """
    Returns true if there exists a directed path from startnode to endnode.
    False otherwise.
    """
    try:
        if nx.bidirectional_dijkstra(DAG, startnode, endnode):
            return True
        else:
            return False
    except:
        return False


def addneighbours(
    i, j, queue, partial_S1, partial_S2, S12, distance_traveled, distance, dx, dy
):
    """
    INPUT:	S  = MxN matrix, where each cell is either 0 or 1.
            dx = grid spacing in the x direction (i.e. the direction with N cells)
            dy = grid spacing in the x direction (i.e. the direction with M cells)

    OUTPUT:	MxN matrix, where each cell contains the shortest distance from that particular cell to a cell with value 0.
    """
    # S     #N    #W    #E    #SW   #SE   #NW   #NE
    ivals = [i - 1, i + 1, i, i, i - 1, i - 1, i + 1, i + 1]
    jvals = [j, j, j - 1, j + 1, j - 1, j + 1, j - 1, j + 1]

    for k in range(len(ivals)):
        append = True

        if (
            partial_S1[ivals[k], jvals[k]] == 0
            and partial_S2[ivals[k], jvals[k]] == 0
            and S12[ivals[k], jvals[k]] == 0
        ):  # Check that point is in active region
            append = False

        if append:
            dl = np.sqrt(((j - jvals[k]) * dx) ** 2 + ((i - ivals[k]) * dy) ** 2)
            inc = 2 * dl / (distance[i, j] + distance[ivals[k], jvals[k]])

            new_distance = distance_traveled[i, j] + inc
            if new_distance < distance_traveled[ivals[k], jvals[k]]:
                queue.append(
                    {
                        "i": ivals[k],
                        "j": jvals[k],
                        "parenti": i,
                        "parentj": j,
                        "traveled": new_distance,
                    }
                )


def shortest_distance_paths(
    partial_S1, partial_S2, S1, S2, amp1, amp2, dx, dy, distance, amplitude_weighted
):
    """
    INPUT:	S  = MxN matrix, where each cell is either 0 or 1.
            dx = grid spacing in the x direction (i.e. the direction with N cells)
            dy = grid spacing in the x direction (i.e. the direction with M cells)

    OUTPUT:	MxN matrix, where each cell contains the shortest distance from that particular cell to a cell with value 0.
    """

    [M, N] = np.shape(S1)
    M = int(M)
    N = int(N)

    S12 = np.zeros((M, N))
    S12[S1 + S2 == 1] = 1

    if amplitude_weighted:
        for i in range(M):
            for j in range(N):
                if S1[i, j] == 1 or S2[i, j] == 1:
                    distance[i, j] *= np.max([amp1[i, j], amp2[i, j]])
    distance += 0.01 * np.sqrt(dx * dy)

    ###################################
    # CALCULATE INITIAL OPTIMAL PATHS #
    ###################################

    distance_traveled = (
        np.ones((M, N)) * np.inf
    )  # book keeping of the non-Euclidean distance traveled to each cell.
    distance_traveled[
        partial_S1 == 1
    ] = 0  # zero distance for all cells on the front of S1.

    parent_x = np.ones((M, N), dtype=int) * (-1)
    parent_y = np.ones((M, N), dtype=int) * (-1)

    queue = []

    for i in range(M)[1:-2]:
        for j in range(N)[1:-2]:
            if partial_S1[i, j] == 1:
                addneighbours(
                    i,
                    j,
                    queue,
                    partial_S1,
                    partial_S2,
                    S12,
                    distance_traveled,
                    distance,
                    dx,
                    dy,
                )  # add all neighbours to cells on the front of S1.
    while queue:
        queue.sort(key=itemgetter("traveled"))
        reached_cell = queue.pop(0)

        i = reached_cell["i"]
        j = reached_cell["j"]

        if reached_cell["traveled"] < distance_traveled[i, j]:
            distance_traveled[i, j] = reached_cell["traveled"]
            addneighbours(
                i,
                j,
                queue,
                partial_S1,
                partial_S2,
                S12,
                distance_traveled,
                distance,
                dx,
                dy,
            )
            parent_y[i, j] = reached_cell["parenti"]
            parent_x[i, j] = reached_cell["parentj"]

        print("Calculating optimal paths:" + str(len(queue)) + 10 * " ", end="\r")

    DAG = nx.DiGraph()

    DAG.add_nodes_from(range(M * N))

    already_reached = np.zeros((M, N), dtype=int)

    for i in range(M):
        for j in range(N):
            if parent_x[i, j] != -1:
                parent_index = parent_y[i, j] * N + parent_x[i, j]
                child_index = i * N + j
                already_reached[i, j] = 1
                DAG.add_edge(int(parent_index), int(child_index))

    #########################
    # CALCULATE "HELP" MAPS #
    #########################

    children_map = np.zeros((M, N), dtype=int)
    for i in range(M)[1:-1]:
        for j in range(N)[1:-1]:
            pos_i = parent_y[i, j]
            pos_j = parent_x[i, j]

            children_map[pos_i, pos_j] = 1

    without_children = np.zeros((M, N), dtype=int)
    for i in range(M)[1:-1]:
        for j in range(N)[1:-1]:
            if children_map[i, j] == 0 and S12[i, j] == 1 and partial_S2[i, j] == 0:
                without_children[i, j] = 1

    source = np.ones((M, N), dtype=int) * (-1)
    distance_traveled_inverse = np.ones((M, N)) * np.inf
    distance_traveled_inverse[partial_S2 == 1] = 0
    for i in range(M)[1:-1]:
        for j in range(N)[1:-1]:
            if partial_S2[i, j] == 1:

                source[i, j] = i * N + j

                temp_i = i
                temp_j = j
                pos_i = parent_y[i, j]
                pos_j = parent_x[i, j]

                while pos_i != -1:

                    dl = np.sqrt(
                        ((pos_j - temp_j) * dx) ** 2 + ((pos_i - temp_i) * dy) ** 2
                    )
                    inc = 2 * dl / (distance[pos_i, pos_j] + distance[temp_i, temp_j])

                    if (
                        distance_traveled_inverse[pos_i, pos_j]
                        > distance_traveled_inverse[temp_i, temp_j] + inc
                    ):
                        distance_traveled_inverse[pos_i, pos_j] = (
                            distance_traveled_inverse[temp_i, temp_j] + inc
                        )
                        source[pos_i, pos_j] = i * N + j

                    temp_i = pos_i
                    temp_j = pos_j
                    pos_i = parent_y[temp_i, temp_j]
                    pos_j = parent_x[temp_i, temp_j]

    ##########################################
    # CALCULATE ADDITIONAL NON-OPTIMAL PATHS #
    ##########################################

    additional_edges = []
    queue = []

    for i in range(M)[1:-2]:
        for j in range(N)[1:-2]:
            parent_penalty = np.copy(distance_traveled_inverse)
            parent_penalty[parent_y[i, j], parent_x[i, j]] = np.inf

            if (
                without_children[i, j] == 1
                and np.min(np.min(parent_penalty[i - 1 : i + 2, j - 1 : j + 2]))
                < np.inf
            ):
                for index in range(9):
                    [delta_i, delta_j] = np.unravel_index(
                        np.argmin(parent_penalty[i - 1 : i + 2, j - 1 : j + 2]), (3, 3)
                    )

                    childi = i + delta_i - 1
                    childj = j + delta_j - 1

                    if already_reached[childi, childj]:
                        continue

                    if directed_path_exists(
                        DAG, childi * N + childj, i * N + j
                    ):  # Check if directed path exists in opposite direction.
                        parent_penalty[childi, childj] = np.inf
                        continue

                    if not distance_traveled_inverse[childi, childj] < np.inf:
                        continue  ## This is new - before we passed

                    dl = np.sqrt(((j - childj) * dx) ** 2 + ((i - childi) * dy) ** 2)
                    inc = 2 * dl / (distance[i, j] + distance[childi, childj])

                    min_dist = (
                        np.min(np.min(parent_penalty[i - 1 : i + 2, j - 1 : j + 2]))
                        + inc
                    )

                    already_reached[childi, childj] = 1
                    queue.append(
                        {
                            "childi": childi,
                            "childj": childj,
                            "parenti": i,
                            "parentj": j,
                            "dist": min_dist,
                        }
                    )

                    break

    # TODO: This is slow! Improve performance.
    while queue:
        queue.sort(key=itemgetter("dist"))
        new_cell = queue.pop(0)

        i = new_cell["parenti"]
        j = new_cell["parentj"]

        if without_children[i, j] == 1 and new_cell["dist"] < np.inf:
            childi = new_cell["childi"]
            childj = new_cell["childj"]

            parent_index = i * N + j
            child_index = childi * N + childj

            if directed_path_exists(DAG, child_index, parent_index):
                continue

            without_children[i, j] = 0
            DAG.add_edge(int(parent_index), int(child_index))
            additional_edges.append((i, j, childi, childj))

            source[i, j] = source[childi, childj]

            dl = np.sqrt(((j - childj) * dx) ** 2 + ((i - childi) * dy) ** 2)
            inc = 2 * dl / (distance[i, j] + distance[childi, childj])

            distance_traveled_inverse[i, j] = new_cell["dist"] + inc

            for delta_i in [-1, 0, 1]:
                for delta_j in [-1, 0, 1]:
                    parent_penalty = np.copy(distance_traveled_inverse)
                    parent_penalty[
                        parent_y[i + delta_i, j + delta_j],
                        parent_x[i + delta_i, j + delta_j],
                    ] = np.inf

                    if (
                        without_children[i + delta_i, j + delta_j] == 1
                        and np.min(
                            np.min(
                                parent_penalty[
                                    i + delta_i - 1 : i + delta_i + 2,
                                    j + delta_j - 1 : j + delta_j + 2,
                                ]
                            )
                        )
                        < np.inf
                    ):
                        for index in range(9):
                            [delta_i2, delta_j2] = np.unravel_index(
                                np.argmin(
                                    parent_penalty[
                                        i + delta_i - 1 : i + delta_i + 2,
                                        j + delta_j - 1 : j + delta_j + 2,
                                    ]
                                ),
                                (3, 3),
                            )
                            childi = i + delta_i + delta_i2 - 1
                            childj = j + delta_j + delta_j2 - 1

                            if already_reached[childi, childj]:
                                continue

                            if directed_path_exists(
                                DAG,
                                childi * N + childj,
                                (i + delta_i) * N + j + delta_j,
                            ):
                                parent_penalty[childi, childj] = np.inf
                                continue

                            dl = np.sqrt(
                                ((j + delta_j - childj) * dx) ** 2
                                + ((i + delta_i - childi) * dy) ** 2
                            )
                            inc = (
                                2
                                * dl
                                / (
                                    distance[i + delta_i, j + delta_j]
                                    + distance[childi, childj]
                                )
                            )

                            min_dist = (
                                np.min(
                                    np.min(
                                        parent_penalty[
                                            i + delta_i - 1 : i + delta_i + 2,
                                            j + delta_j - 1 : j + delta_j + 2,
                                        ]
                                    )
                                )
                                + inc
                            )

                            if without_children[i + delta_i, j + delta_j] == 1:
                                already_reached[childi, childj] = 1
                                queue.append(
                                    {
                                        "childi": childi,
                                        "childj": childj,
                                        "parenti": i + delta_i,
                                        "parentj": j + delta_j,
                                        "dist": min_dist,
                                    }
                                )

            temp_i = i
            temp_j = j
            pos_i = parent_y[temp_i, temp_j]
            pos_j = parent_x[temp_i, temp_j]

            while pos_i != -1:
                dl = np.sqrt(
                    ((pos_j - temp_j) * dx) ** 2 + ((pos_i - temp_i) * dy) ** 2
                )
                inc = 2 * dl / (distance[pos_i, pos_j] + distance[temp_i, temp_j])

                if (
                    distance_traveled_inverse[pos_i, pos_j]
                    > distance_traveled_inverse[temp_i, temp_j] + inc
                ):
                    distance_traveled_inverse[pos_i, pos_j] = (
                        distance_traveled_inverse[temp_i, temp_j] + inc
                    )
                    source[pos_i, pos_j] = source[temp_i, temp_j]

                for delta_i in [-1, 0, 1]:
                    for delta_j in [-1, 0, 1]:
                        parent_penalty = np.copy(distance_traveled_inverse)
                        parent_penalty[
                            parent_y[pos_i + delta_i, pos_j + delta_j],
                            parent_x[pos_i + delta_i, pos_j + delta_j],
                        ] = np.inf

                        if (
                            without_children[pos_i + delta_i, pos_j + delta_j] == 1
                            and np.min(
                                np.min(
                                    parent_penalty[
                                        pos_i + delta_i - 1 : pos_i + delta_i + 2,
                                        pos_j + delta_j - 1 : pos_j + delta_j + 2,
                                    ]
                                )
                            )
                            < np.inf
                        ):
                            for index in range(9):
                                [delta_i2, delta_j2] = np.unravel_index(
                                    np.argmin(
                                        parent_penalty[
                                            pos_i + delta_i - 1 : pos_i + delta_i + 2,
                                            pos_j + delta_j - 1 : pos_j + delta_j + 2,
                                        ]
                                    ),
                                    (3, 3),
                                )
                                childi = pos_i + delta_i + delta_i2 - 1
                                childj = pos_j + delta_j + delta_j2 - 1

                                if directed_path_exists(
                                    DAG,
                                    childi * N + childj,
                                    (pos_i + delta_i) * N + pos_j + delta_j,
                                ):
                                    parent_penalty[childi, childj] = np.inf
                                    continue

                                dl = np.sqrt(
                                    ((pos_j + delta_j - childj) * dx) ** 2
                                    + ((pos_i + delta_i - childi) * dy) ** 2
                                )
                                inc = (
                                    2
                                    * dl
                                    / (
                                        distance[pos_i + delta_i, pos_j + delta_j]
                                        + distance[childi, childj]
                                    )
                                )

                                min_dist = (
                                    np.min(
                                        np.min(
                                            parent_penalty[
                                                pos_i
                                                + delta_i
                                                - 1 : pos_i
                                                + delta_i
                                                + 2,
                                                pos_j
                                                + delta_j
                                                - 1 : pos_j
                                                + delta_j
                                                + 2,
                                            ]
                                        )
                                    )
                                    + inc
                                )

                                if (
                                    without_children[pos_i + delta_i, pos_j + delta_j]
                                    == 1
                                ):
                                    queue.append(
                                        {
                                            "childi": childi,
                                            "childj": childj,
                                            "parenti": pos_i + delta_i,
                                            "parentj": pos_j + delta_j,
                                            "dist": min_dist,
                                        }
                                    )

                temp_i = pos_i
                temp_j = pos_j
                pos_i = parent_y[temp_i, temp_j]
                pos_j = parent_x[temp_i, temp_j]

        print(
            "Calculating non-optimal paths - ramining queue:"
            + str(len(queue))
            + 10 * " ",
            end="\r",
        )

    if not nx.is_directed_acyclic_graph(DAG):
        raise RuntimeError("The graph is not a DAG!")

    print("Finished calculating shortest distance paths...")
    return DAG, source, parent_x, parent_y, additional_edges
