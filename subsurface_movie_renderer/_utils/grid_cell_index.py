#!/usr/bin/env python
# encoding: utf-8

# Author: Anders Fredrik Kiaer, Oct. 8th 2011

import numpy as np


def grid_cell_index(input_amp_maps):
    """
    INPUT:	amp_maps = list of amplitude maps, i.e. MxN matrices. Interpreted values are > 0.

    OUTPUT:	the index vector
    """

    amp_maps = []
    for input_amp_map in input_amp_maps:
        amp_maps.append(np.copy(input_amp_map))

    for amp_map in amp_maps:
        amp_map[amp_map <= 0] = 0
        amp_map[amp_map > 0] = 1

    S = len(amp_maps)
    [M, N] = np.shape(amp_maps[0])
    index_vector = list(range((S - 1) * M * N))
    been_different = np.zeros((M, N))
    counter = -1

    # Add all grid cells during the first interpolation interval
    for i in range(M):
        for j in range(N):
            counter += 1
            index_vector[i * N + j] = counter

    # Add new grid cells if they they are going back to a previous value.
    for k in range(S - 1)[1:]:
        prev = amp_maps[k - 1]
        amp_map = amp_maps[k]
        for i in range(M):
            for j in range(N):
                if been_different[i, j] == 1 and amp_map[i, j] != prev[i, j]:
                    counter += 1
                    index_vector[i * N + j + k * M * N] = counter
                elif amp_map[i, j] != prev[i, j]:
                    index_vector[i * N + j + k * M * N] = index_vector[
                        i * N + j + (k - 1) * M * N
                    ]
                    been_different[i, j] = 1
                else:
                    index_vector[i * N + j + k * M * N] = index_vector[
                        i * N + j + (k - 1) * M * N
                    ]

    return (counter + 1), index_vector
