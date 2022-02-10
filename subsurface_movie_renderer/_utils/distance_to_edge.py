import numpy as np


def distance_to_edge(S, dx, dy):
    """
    INPUT:	S  = MxN matrix, where each cell is either 0 or 1.
            dx = grid spacing in the x direction (i.e. the direction with N cells)
            dy = grid spacing in the y direction (i.e. the direction with M cells)

    OUTPUT:	MxN matrix, where each cell contains the shortest distance from the center of that particular cell to the center of a cell with value 0. Units are the same as for dx and dy.
    """

    if np.sum(np.sum(S == 0)) == 0:
        raise ValueError("The matrix S must have at least one cell with value 0.")

    [M, N] = np.shape(S)

    x = np.linspace(0, N - 1, N) * dx
    y = np.linspace(0, M - 1, M) * dy

    [X, Y] = np.meshgrid(x, y)

    DISTANCE = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if S[i, j] == 1:
                DIST = (X - X[i, j]) ** 2 + (Y - Y[i, j]) ** 2
                DISTANCE[i, j] = np.min(np.min(DIST[S == 0]))

        print(str((i + 1) * 100 / M) + "%" + 10 * " ", end="\r")

    print("Finished calculating distance map...")

    return np.sqrt(DISTANCE)
