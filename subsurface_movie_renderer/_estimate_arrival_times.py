import os
import json
import pickle
from pathlib import Path

from cvxopt import matrix, solvers, spmatrix, sparse
import networkx as nx
import numpy as np
from scipy.sparse import identity, lil_matrix  # sparse matrix library
from scipy.sparse.linalg import spsolve  # sparse matrix solver

from ._utils.distance_to_edge import distance_to_edge  #
from ._utils.shortest_distance_paths import shortest_distance_paths  #
from ._utils.grid_cell_index import grid_cell_index  #
from ._utils.find_topology_changes import find_topology_changes
from ._utils.fill_sparse_matrix import fill_sparse_matrix


def estimate_arrival_times(
    data_folder, horizon, start_time, surveys_metadata, tmp_dir_path
):

    global_surveys = list(surveys_metadata.keys())
    global_times = [survey["time"] for survey in surveys_metadata.values()]

    lamb_T = 1e2  # Tikhonov regularization parameter	# intra slowness
    lamb_S = 1e2  # Tikhonov regularization parameter	# inter slowness
    gamma_L = 0.8
    gamma_U = 1.2

    # possible values: 'identity','first_derivative','second_derivative'
    regularization = "identity"

    X = np.load(os.path.join(data_folder, "X_coordinates.npy"))
    Y = np.load(os.path.join(data_folder, "Y_coordinates.npy"))

    dx = np.mean(np.diff(X[0, :]))  # grid spacing, x direction [m]
    dy = np.mean(np.diff(Y[:, 0]))  # grid spacing, y direction [m]

    [M, N] = np.shape(X)

    M = int(M)
    N = int(N)

    # ---------- LOAD ALL AMPLITUDE MAPS INTO MEMORY -----

    if start_time >= global_times[0]:
        raise ValueError(
            "Start time for horizon {horizon} needs to be before first seismic survey."
        )

    surveys = ["horizon_starts"]
    times = [start_time]
    amp_maps = [np.zeros((M, N))]

    for i, survey in enumerate(global_surveys):
        try:
            amp_maps.append(
                np.load(os.path.join(data_folder, f"{horizon}--{survey}.npy"))
                * surveys_metadata[survey]["scaling"]
            )
            surveys.append(survey)
            times.append(global_times[i])
        except FileNotFoundError:
            print(f"Could not find data corresponding to survey {survey}")

    S = len(surveys)

    [N_v, index_vector] = grid_cell_index(
        amp_maps
    )  # calculate index vector and number of independent slowness values

    # --------------- START MAIN PART ---------------------

    # velocity mapping:
    # 	i = (y-coordinate: 0, 1, ..., M-1)
    # 	j = (x-coordinate: 0, 1, ..., N-1)
    # 	k = (interpolation interval: 0, 1, ..., S-2)
    #
    # 	n = i*N + j + k*M*N

    DELTA_T = matrix(0.0, (N_v, 1))

    P = spmatrix(0.0, [0], [0], (N_v, N_v))
    C = spmatrix(0.0, [0], [0], (N_v, N_v))

    equation_index = -1
    equation_index_new_values = -1

    D_t = spmatrix(
        0.0, [0], [0], (N_v, N_v)
    )  # First order derivative in time		# NOT FULL RANK!

    D_x = spmatrix(
        0.0, [0], [0], (N_v, N_v)
    )  # First order derivative in x direction		# NOT FULL RANK!
    D_y = spmatrix(
        0.0, [0], [0], (N_v, N_v)
    )  # First order derivative in y direction		# NOT FULL RANK!

    D_xx = spmatrix(
        0.0, [0], [0], (N_v, N_v)
    )  # Second order derivative in x direction	# NOT FULL RANK!
    D_yy = spmatrix(
        0.0, [0], [0], (N_v, N_v)
    )  # Second order derivative in y direction	# NOT FULL RANK!

    w_lower = matrix(-1e-5, (N_v, 1))  # lower bound on slowness values
    w_upper = matrix(-1e-5, (N_v, 1))  # lower bound on slowness values

    w_lower_top = matrix(-1e-5, (N_v, 1))  # lower bound on slowness values
    w_upper_top = matrix(-1e-5, (N_v, 1))  # lower bound on slowness values

    topology_change_recordings = []

    for k in range(S - 1):
        print(
            f"Started on interpolation over the interval ({times[k]:.2f}, {times[k+1]:.2f})..."
        )

        S1 = np.copy(amp_maps[k])
        S2 = np.copy(amp_maps[k + 1])

        t_star = times[k + 1] - times[k]

        S1[S1 <= 0] = 0
        S1[S1 > 0] = 1

        S2[S2 <= 0] = 0
        S2[S2 > 0] = 1

        topology_changes = find_topology_changes(S1, S2, dx, dy)

        # Insert the calculated "feeding points" by changing the value of S1 in that particular cell:
        topology_changes_singleindex = []
        for index in topology_changes:
            topology_changes_singleindex.append(index[0] * N + index[1])
            S1[index[0], index[1]] = 1 - S1[index[0], index[1]]

        S12 = np.zeros((M, N))
        S12[S1 + S2 == 1] = 1

        partial_S1 = np.zeros((M, N))
        partial_S2 = np.zeros((M, N))

        for i in range(M)[1:-2]:
            for j in range(N)[1:-2]:
                # print i, M, j, N
                if (
                    S1[i, j] == 1
                    and np.sum(np.sum(S1[i - 1 : i + 2, j - 1 : j + 2])) < 9
                ):  # on S1 boundary
                    partial_S1[i, j] = 1
                if (
                    S2[i, j] == 1
                    and np.sum(np.sum(S2[i - 1 : i + 2, j - 1 : j + 2])) < 9
                ):  # on S2 boundary
                    partial_S2[i, j] = 1

        distance = distance_to_edge(S12, dx, dy)  # calculate distance map to edge
        np.savez(
            tmp_dir_path
            / f"distance_maps--{horizon}--{surveys[k+1]}--{surveys[k]}.npz",
            distance=distance,
        )

        [DAG, source, parent_x, parent_y, additional_edges] = shortest_distance_paths(
            partial_S1,
            partial_S2,
            S1,
            S2,
            amp_maps[k],
            amp_maps[k + 1],
            dx,
            dy,
            distance,
            True,
        )  # parent map

        FILE = open(
            tmp_dir_path / f"DAG_{horizon}_{surveys[k + 1]}_{surveys[k]}.dat",
            "wb",
        )
        pickle.dump(DAG, FILE)
        FILE.close()

        ################################
        ## Calculate path expressions ##
        ################################

        boundary_points = np.zeros((M, N))
        traveled_from_boundary = np.zeros((M, N))
        path_end_point = np.zeros((M, N))

        [
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
        ] = fill_sparse_matrix(
            N_v,
            equation_index,
            partial_S1,
            DAG,
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
        )

        equation_index += 1

        #######################################
        ## Calculate first order derivatives ##
        #######################################

        if regularization == "first_derivative":
            for i in range(M):
                for j in range(N)[1:]:
                    D_x[
                        index_vector[int(i * N + j + k * M * N)],
                        index_vector[int(i * N + j + k * M * N)],
                    ] = 1.0
                    D_x[
                        index_vector[int(i * N + j + k * M * N)],
                        index_vector[int(i * N + (j - 1) + k * M * N)],
                    ] = -1.0
            for i in range(M)[1:]:
                for j in range(N):
                    D_y[
                        index_vector[int((i * N + j) + k * M * N)],
                        index_vector[int((i * N + j) + k * M * N)],
                    ] = 1.0
                    D_y[
                        index_vector[int((i * N + j) + k * M * N)],
                        index_vector[int(((i - 1) * N + j) + k * M * N)],
                    ] = -1.0

        ########################################
        ## Calculate second order derivatives ##
        ########################################

        if regularization == "second_derivative":
            for i in range(M):
                for j in range(N)[1:-1]:
                    D_xx[
                        index_vector[int((i * N + j) + k * M * N)],
                        index_vector[int((i * N + j + 1) + k * M * N)],
                    ] = 1.0
                    D_xx[
                        index_vector[int((i * N + j) + k * M * N)],
                        index_vector[int((i * N + j) + k * M * N)],
                    ] = -2.0
                    D_xx[
                        index_vector[int((i * N + j) + k * M * N)],
                        index_vector[int((i * N + j - 1) + k * M * N)],
                    ] = 1.0

            for i in range(M)[1:-1]:
                for j in range(N):
                    D_yy[
                        index_vector[int((i * N + j) + k * M * N)],
                        index_vector[int(((i + 1) * N + j) + k * M * N)],
                    ] = 1.0
                    D_yy[
                        index_vector[int((i * N + j) + k * M * N)],
                        index_vector[int((i * N + j) + k * M * N)],
                    ] = -2.0
                    D_yy[
                        index_vector[int((i * N + j) + k * M * N)],
                        index_vector[int(((i - 1) * N + j) + k * M * N)],
                    ] = 1.0

        #################################
        ## Calculate "time derivative" ##
        #################################

        if k > 0:
            for i in range(M):
                for j in range(N):
                    if (
                        partial_S1[i, j] == 1
                        and index_vector[int((i * N + j) + k * M * N)]
                        != index_vector[int((i * N + j) + (k - 1) * M * N)]
                    ):
                        equation_index_new_values += 1
                        D_t[
                            equation_index_new_values,
                            index_vector[int((i * N + j) + k * M * N)],
                        ] = 1.0
                        D_t[
                            equation_index_new_values,
                            index_vector[int((i * N + j) + (k - 1) * M * N)],
                        ] = -1.0

    # Remove parts of P and C not in use:

    P = P[: equation_index + 1, :]
    C = C[: equation_index + 1, :]

    # Calculate mean slowness value:

    w_mean = 0
    count = 0

    maxval = -1000
    for i in range(N_v):
        if w_lower[i, 0] > 0:
            w_mean += w_lower[i, 0]
            count += 1

        if w_lower[i, 0] > maxval:
            maxval = w_lower[i, 0]

    print()
    print("count", count)
    w_mean /= 1  # count   # THIS FAILS BECAUSE count is 0. Investigate.

    top_change_final = []
    for top_change in topology_change_recordings:
        current = 0

        for j in top_change[2]:
            current += w_lower_top[j, 0]

        if current == 0:
            continue

        t_new = w_mean * len(top_change[2]) / current

        if t_new > top_change[3]:
            t_new = top_change[3]

        for j in top_change[2]:
            # FINN UT HVILKE CELLER SOM ALLEREDE HAR VERDI HER.
            if w_lower[j, 0] < 0 or w_lower[j, 0] > w_lower_top[j, 0] * t_new:
                w_lower[j, 0] = w_lower_top[j, 0] * t_new
            if w_upper[j, 0] < 0 or w_upper[j, 0] < w_upper_top[j, 0] * t_new:
                w_upper[j, 0] = w_upper_top[j, 0] * t_new

        DELTA_T[top_change[0] : top_change[1], 0] = t_new

        top_change_final.append(
            [top_change[4], top_change[5], top_change[6], t_new]
        )  # k, i, j (topology source indices)

    print("Create sparse identity matrix, " + str(N_v) + "x" + str(N_v))

    I = spmatrix(1.0, range(N_v), range(N_v))

    if regularization == "identity":
        A = sparse([P, lamb_T * I])
    elif regularization == "first_derivative":
        A = sparse([P, lamb_T * D_x, lamb_T * D_y])
    elif regularization == "second_derivative":
        A = sparse([P, lamb_T * D_xx, lamb_T * D_yy])
    else:
        raise ValueError("Unknown regularization parameter.")

    if equation_index_new_values >= 0:
        D_t = D_t[: equation_index_new_values + 1, :]
        A = sparse([A, lamb_S * D_t])
        print("Added time derivative")

    [m, n] = A.size
    b = matrix(0.0, (m, 1))
    b[: equation_index + 1, 0] = DELTA_T[: equation_index + 1, 0]

    for i in range(N_v):
        if w_lower[i, 0] < 0:
            w_lower[i, 0] = 0.0

    for i in range(N_v):
        if w_upper[i, 0] < 0:
            w_upper[i, 0] = 1000.0

    C = sparse([I, -I, -C])
    d = matrix([w_lower, -w_upper, -DELTA_T[: equation_index + 1, 0]])

    ### MINIMIZE:	||Ax-b||^2
    ### SUBJECT TO:	Cx >= d

    print("Started solving equation system...")
    x = solvers.coneqp(A.T * A, -A.T * b, -C, -d)["x"]
    print(f"Finished solving arrival time equation system for horizon {horizon}...")

    w = np.array(x)  # Convert to native numpy array from cvxopt array

    for k in range(S - 1):
        w_collapsed = np.zeros((M, N))
        for i in range(M):  # calculate Tikhonov regularization matrices
            for j in range(N):
                w_collapsed[i, j] = w[index_vector[int((i * N + j) + k * M * N)]]

    ######################################################################
    ###### CALCULATE ARRIVAL TIMES BASED ON CALCULATED SLOWNESSES ########
    ######################################################################

    for k in range(S - 1):
        S1 = np.copy(amp_maps[k])
        S2 = np.copy(amp_maps[k + 1])

        S1[S1 <= 0] = 0
        S1[S1 > 0] = 1

        S2[S2 <= 0] = 0
        S2[S2 > 0] = 1

        t_max = times[k + 1] - times[k]

        data = np.load(
            tmp_dir_path / f"distance_maps--{horizon}--{surveys[k+1]}--{surveys[k]}.npz"
        )
        distance = data["distance"]

        FILE = open(
            tmp_dir_path / f"DAG_{horizon}_{surveys[k + 1]}_{surveys[k]}.dat",
            "rb",
        )
        DAG = pickle.load(FILE)
        FILE.close()

        # FORWARD PROPAGATION

        forward_propagation = lil_matrix((M * N, M * N))
        time_diff_forward = np.zeros((M * N, 1))

        for i in range(M):
            for j in range(N):
                predecessors = list(DAG.predecessors(i * N + j))
                forward_propagation[i * N + j, i * N + j] = 1

                if predecessors != []:
                    for index in range(len(predecessors)):
                        parent_i = predecessors[index] // N
                        parent_j = predecessors[index] % N

                        d = np.sqrt(
                            np.abs(parent_i - i) * dy ** 2
                            + np.abs(parent_j - j) * dx ** 2
                        )

                        forward_propagation[
                            i * N + j, parent_i * N + parent_j
                        ] = -1.0 / len(predecessors)
                        time_diff_forward[i * N + j, 0] = (
                            d
                            * (
                                w[index_vector[int((i * N + j) + k * M * N)]]
                                + w[
                                    index_vector[int((predecessors[index]) + k * M * N)]
                                ]
                            )
                            / (2.0 * len(predecessors))
                        )
                else:
                    if S1[i, j] == 1 and S2[i, j] == 0:
                        time_diff_forward[i * N + j, 0] = (
                            w[index_vector[int((i * N + j) + k * M * N)]] * dx / 2.0
                        )
                    else:
                        time_diff_forward[i * N + j, 0] = 0

        for top_change in top_change_final:
            if top_change[0] == k:
                time_diff_forward[top_change[1] * N + top_change[2]] = (
                    t_max - top_change[3]
                )

        forward_propagation = forward_propagation.tocsr()

        AT = spsolve(forward_propagation, time_diff_forward)
        AT = np.reshape(AT, (M, N))

        AT *= 100 / t_max
        AT[AT > 100] = 100

        np.savez(
            tmp_dir_path / f"{horizon}_{k}.npz",
            AT=AT,
            X=X,
            Y=Y,
            AMP1=amp_maps[k],
            AMP2=amp_maps[k + 1],
        )

    (tmp_dir_path / f"{horizon}_metadata.json").write_text(json.dumps(times))
