import numpy as np


def return_paths(dag, start_node):

    queue = []
    path = []
    paths = []

    node = start_node
    path.append(int(start_node))

    level = 1

    still_continue = True

    while still_continue:
        successors = list(dag.successors(node))

        if successors:
            if level > len(queue):
                queue.append(successors)
                node = successors[0]
            else:
                node = queue[level - 1][0]

            path.append(int(node))
            level = level + 1

        if not successors:
            paths.append(path)

            while len(queue) > 0 and len(queue[-1]) < 2:
                queue.pop()

            node = start_node
            path = [int(start_node)]

            level = 1

            maxchoices = 0
            for queue_element in queue:
                if len(queue_element) >= maxchoices:
                    maxchoices = len(queue_element)

            if maxchoices == 0:
                still_continue = False
            else:
                queue[level - 2].pop(0)

    return paths


# TODO: Remove these disables
# pylint: disable=too-many-arguments, too-many-locals, too-many-nested-blocks
# pylint: disable=too-many-branches, too-many-statements, unused-argument
def fill_sparse_matrix(
    N_v,
    equation_index,
    partial_S1,
    dag,
    topology_changes_singleindex,
    topology_change_recordings,
    boundary_points,
    t_star,
    k,
    DELTA_T,
    dx,
    dy,
    P,
    C,
    index_vector,
    traveled_from_boundary,
    gamma_L,
    gamma_U,
    w_lower_top,
    w_upper_top,
    w_lower,
    w_upper,
):

    [M, N] = np.shape(partial_S1)

    # DELTA_T = matrix(0.0, (N_v,1))

    # P = spmatrix(0.0, [0], [0], (N_v,N_v))
    # C = spmatrix(0.0, [0], [0], (N_v,N_v))

    # w_lower = (-1e-5)*np.ones(N_v, dtype=np.double)	# lower bound on slowness values
    # w_upper = (-1e-5)*np.ones(N_v, dtype=np.double)	# lower bound on slowness values

    # w_lower_top = (-1e-5)*np.ones(N_v, dtype=np.double)# lower bound on slowness values
    # w_upper_top = (-1e-5)*np.ones(N_v, dtype=np.double)# lower bound on slowness values

    for i in range(M):
        for j in range(N):
            if partial_S1[i, j] == 1 and not list(dag.predecessors(i * N + j)):
                if topology_changes_singleindex.count(i * N + j) > 0:
                    topology_change_source = True
                    topology_change_recordings.append(
                        [equation_index + 1, -1, set(), t_star, k, i, j]
                    )
                else:
                    topology_change_source = False

                paths = return_paths(dag, i * N + j)

                boundary_points[i, j] = 1

                numberpaths = len(paths)
                counter = 0
                for path in paths:
                    counter += 1
                    if len(path) <= 2:
                        continue

                    total_length = 0

                    pos_i = path[-1] // N
                    pos_j = path[-1] % N

                    equation_index = equation_index + 1

                    DELTA_T[equation_index] = t_star

                    for index in range(len(path))[1:]:
                        parent_i = path[index - 1] // N
                        parent_j = path[index - 1] % N

                        pos_i = path[index] // N
                        pos_j = path[index] % N

                        dist = np.sqrt(
                            abs(pos_i - parent_i) * dy * dy
                            + abs(pos_j - parent_j) * dx * dx
                        )
                        total_length = total_length + dist

                        P[
                            equation_index, index_vector[path[index - 1] + k * M * N]
                        ] += (dist / 2.0)
                        P[equation_index, index_vector[path[index] + k * M * N]] += (
                            dist / 2.0
                        )

                        C[
                            equation_index, index_vector[path[index - 1] + k * M * N]
                        ] += (dist / 2.0)
                        C[equation_index, index_vector[path[index] + k * M * N]] += (
                            dist / 2.0
                        )

                        traveled_from_boundary[parent_i, parent_j] = 1
                        traveled_from_boundary[pos_i, pos_j] = 1

                    if topology_change_source:
                        new_LB = 1.0 / total_length
                        new_UB = 1.0 / total_length
                    else:
                        new_LB = gamma_L * t_star / total_length
                        new_UB = gamma_U * t_star / total_length

                    for path_element in path:
                        pos_i = path_element // N
                        pos_j = path_element % N

                        if topology_change_source:
                            topology_change_recordings[-1][2].add(
                                index_vector[pos_i * N + pos_j + k * M * N]
                            )
                            if (
                                w_lower_top[index_vector[pos_i * N + pos_j + k * M * N]]
                                < 0
                                or new_LB
                                < w_lower_top[
                                    index_vector[pos_i * N + pos_j + k * M * N]
                                ]
                            ):
                                w_lower_top[
                                    index_vector[pos_i * N + pos_j + k * M * N]
                                ] = new_LB
                            if (
                                w_upper_top[index_vector[pos_i * N + pos_j + k * M * N]]
                                < 0
                                or new_UB
                                > w_upper_top[
                                    index_vector[pos_i * N + pos_j + k * M * N]
                                ]
                            ):
                                w_upper_top[
                                    index_vector[pos_i * N + pos_j + k * M * N]
                                ] = new_UB
                        else:
                            if (
                                w_lower[index_vector[pos_i * N + pos_j + k * M * N]] < 0
                                or new_LB
                                < w_lower[index_vector[pos_i * N + pos_j + k * M * N]]
                            ):
                                w_lower[
                                    index_vector[pos_i * N + pos_j + k * M * N]
                                ] = new_LB
                            if (
                                w_upper[index_vector[pos_i * N + pos_j + k * M * N]] < 0
                                or new_UB
                                > w_upper[index_vector[pos_i * N + pos_j + k * M * N]]
                            ):
                                w_upper[
                                    index_vector[pos_i * N + pos_j + k * M * N]
                                ] = new_UB

                    if topology_change_source:
                        topology_change_recordings[-1][1] = equation_index + 1

                    print(
                        str((i * N + j + 1) * 100 / (M * N))
                        + "%, "
                        + str(counter)
                        + "/"
                        + str(numberpaths)
                        + 10 * " ",
                        end="\r",
                    )

    return [
        topology_change_recordings,
        boundary_points,
        equation_index,
        DELTA_T,
        P,
        C,
        traveled_from_boundary,
        w_lower_top,
        w_upper_top,
        w_lower,
        w_upper,
    ]
