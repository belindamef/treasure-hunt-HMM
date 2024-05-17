import os
import numpy as np                                                              # numpy
import more_itertools as it                                                     # iterables


def th_sets(theta, paths):
    """This function generates the state, observation, decision, and action sets
    for the treasure hunt task and agent models

    Inputs
        theta    (obj) : task parameter structure with required fields
            .d   (int) : dimensionality of the square grid world
            .d_s (int) : state vector dimension (agent location, treasure location, hiding spot locations)
            .n_n (int) : number of nodes
            .n_h (int) : number of hiding spots
            .n_s (int) : state space cardinality
        paths    (obj) : paths object storing directory path variables

    Outputs
        NONE

    Saves to disk, if not existing
        S        (arr) : n_s x (2 + n_h) array of state values
        O        (arr) : n_n x 2 array of observation values
        A        (arr) : 5 x 1 array of action values

    Authors - Belinda Fleischmann, Dirk Ostwald
    """
    d     = theta.d                                                             # dimensionality of the square grid world
    d_s   = theta.d_s                                                           # state vector dimension (agent location, treasure location, hiding spot locations)
    n_n   = theta.n_n                                                           # number of nodes
    n_h   = theta.n_h                                                           # number of hiding spots
    n_s   = theta.n_s                                                           # state space cardinality
    nodes = range(1, n_n + 1)                                                   # set of nodes

    # State set
    if not os.path.exists(os.path.join(paths.components, "S.npy")):

        S_raw = np.full([n_s, d_s], np.nan, dtype=np.int8)                      # raw state value array initialization
        S3    = it.distinct_combinations(nodes, r=n_h)                          # distinct combinations of n_h selections of set nodes
        idx   = 0                                                               # row index initialization

        # TODO: evtl. s3 und s2 jeweils + 1, damit Zählung bei 1 und nicht 0 anfängt?
        for s3 in S3:                                                           # hiding spot combination iterations
            for s2 in s3:                                                       # treasure location iterations (possible locations given hiding spots)
                for s1 in nodes:                                                # agent location iterations
                    S_raw[idx, :] = np.hstack(                                  # state value concatenation
                        (np.array([s1, s2]), np.array(list(s3))))               # result in each row [s1, s2, s3]
                    idx = idx + 1                                               # row index update

        S   = np.full([n_s, d_s], np.nan, dtype=np.int8)                        # sorted state value array initialization
        idx = 0                                                                 # row index initialization
        for s1 in nodes:                                                        # agent location iterations
            S_s1 = S_raw[S_raw[:, 0] == s1]                                     # state values with agent location s1
            for s2 in nodes:                                                    # treasure location iterations
                S_s2 = S_s1[S_s1[:, 1] == s2]                                   # state values with agent location s1 and treasure location s2
                S[idx:idx + S_s2.shape[0], :] = S_s2                            # sorted state value array update
                idx = idx + S_s2.shape[0]                                       # row index update

        np.save(os.path.join(paths.components, "S"), S)                         # save to disc

    # Observation set
    O = np.array(
        ([0, 0],                                                                # no treasure on black
         [0, 1],                                                                # no treasure on grey
         [0, 2],                                                                # no treasure on blue
         [1, 0],                                                                # treasure on black
         # [1, 1]  TODO: unmögliche Beobachtung, trotzdem rein nehmen?          # treasure on grey
         [1, 2]),                                                               # treasure on blue
        dtype=int)

    np.save(os.path.join(paths.components, "O"), O)                             # save to disk

    # Action set
    A = np.array([0, -d, 1, d, -1])                                             # actions (drill, north, east, south, west)
    np.save(os.path.join(paths.components, "A"), A)                             # save to disk
