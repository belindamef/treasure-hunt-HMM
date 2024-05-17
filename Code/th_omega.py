import numpy as np
import os
import scipy.sparse as sp
from th_imshow import plot_color_map
from th_helper import humanreadable_time
import time


def th_omega(S, O, theta, paths):
    """This function evaluates the action-dependent and state-conditional
    observation probability distribution of a Bayesian agent for the treasure
    hunt task.

    Inputs:
        theta    (obj) : task parameter structure with required fields
            .d   (int) : dimension of square grid world
            .n_s (int) : state space cardinality
            .n_o (int) : observation space cardinality
        S        (arr) : n_s x 1 + n_h array
        O        (arr) : n_n x 2 array of observation values
        paths    (obj) : paths object storing directory path variables

    Outputs:
        Omega    (arr) : n_s x n_o x 2 observation probability distribution

    """
    # Task parameters and set cardinalities
    d       = theta.d                                                           # dimension of the square grid world
    n_s     = theta.n_s                                                         # state space cardinality (number of states)
    n_o     = theta.n_o                                                         # observation space cardinality (number of observations)
    n_a     = 2                                                                 # compressed action space cardinality (drill/step)

    # Action set
    A = [0, 1]                                                                  # compressed action space (drill/step)

    # Initialize dictionary of observation probability distribution matrices
    Omega = {}
    for p in range(n_a):                                                        # iterate action indices
        Omega[p] = sp.csc_matrix(
            (n_s, n_o),                                                         # shape of Omega[:, :, a]
            dtype=np.int8                                                       # smallest possible datatype: int8
        )

    # Define list of labels for the action-dependent Omega matrices; used for saving or loading matrices from disk
    matrix_names = [
        "Omega_drill",
        "Omega_step"
    ]

    # -----------------------------------------------------------------------------------------------------
    # Compute or load action-dependent observation probability distribution matrices
    # -----------------------------------------------------------------------------------------------------

    for p in range(n_a):                                                        # a_t iterations

        Omega_matrix_name = matrix_names[p]                                     # Get action-specific Matrix label string for path variable

        # this_Omega_p = sp.lil_matrix(
        #     (n_s, n_o),                                                         # shape of Omega[:, :, a]
        #     dtype=np.int8                                                       # smallest possible datatype: int8
        # )
        a = A[p]                                                                # action a \in A (compressed)

        # Compute Omega[p] if not existing on disk
        if not os.path.exists(os.path.join(paths.components, f"{Omega_matrix_name}.npz")):

            # Initialize arrays to store row and col indices, that indicate value 1 entries in Phi[p]; needed to create sparse matrices
            if a == 0:
                n_nonzeros = n_s                                                # number of nonzero values in Omega[p]
            else:
                n_nonzeros = n_s * 2
            rows = np.full((n_nonzeros), np.nan)                                # array to store row indices
            cols = np.full((n_nonzeros), np.nan)                                # array to store col indices

            print(f"Starting evaluation of Omega[{p}]..., matrix_label: {Omega_matrix_name}")

            idx = 0

            for i in range(n_s):                                                # state iterations

                start = time.time()
                for m in range(n_o):                                            # observation iterations

                    s = S[i, :]                                                 # state s \in S
                    o = O[m, :]                                                 # observation o \in O

                    # -------After DRILL actions: ------------------------------
                    if a == 0:                                                  # dril actions (a_t = 0)

                        # CONDITION "DRILL A":           CORRESP MODEL VARIABLE:
                        # ------------------------------------------------------
                        # if new position...                               s[0]
                        # ...(1) IS NOT treasure location                  s[1]
                        # ...(2) IS NOT hiding spot,                       s[2:]
                        # all observation, for which...
                        # ...(3) tr_flag == 0,                             o[0]
                        # ...(4) and node color == 1 (grey),               o[1]

                        if (                                                    # CONDITION "DRILL A"
                                s[0] != s[1]                              # (1)
                                and s[0] not in s[2:]                     # (2)
                                and o[0] == 0                             # (3)
                                and o[1] == 1                             # (4)
                        ):
                            # this_Omega_p[i, m] = 1                              # possible observation
                            rows[idx] = i
                            cols[idx] = m
                            idx += 1

                        # CONDITION "DRILL B":           CORRESP MODEL VARIABLE:
                        # ------------------------------------------------------
                        # if new position...                               s[0]
                        # ...(1) IS NOT treasure location                  s[1]
                        # ...(2) IS hiding spot,                           s[2:]
                        # all observation, for which...
                        # ...(3) tr_flag == 0,                             o[0]
                        # ...(4) and node color == 2 (blue),               o[1]
                        if (                                                    # CONDITION "DRILL B"
                                s[0] != s[1]                              # (1)
                                and s[0] in s[2:]                         # (2)
                                and o[0] == 0                             # (3)
                                and o[1] == 2                             # (4)
                        ):
                            # this_Omega_p[i, m] = 1                              # possible observation
                            rows[idx] = i
                            cols[idx] = m
                            idx += 1

                        # All other observaton probabs remain 0 as initiated.

                # -------After STEP actions: -----------------------------------
                    else:                                                       # step actions (a_t = 1)

                        # CONDITION "STEP A":            CORRESP MODEL VARIABLE:
                        # ------------------------------------------------------
                        # if new position...                               s[0]
                        # ...(1) IS NOT treasure location                  s[1]
                        # ...(2) IS NOT hiding spot,                       s[2:]
                        # all observation, for which...
                        # ...(3) tr_flag == 0,                             o[0]
                        # ...(4) and node color in [0, 1](black or grey),  o[1]
                        if (                                                    # CONDTION "STEP A"
                                s[0] != s[1]                              # (1)
                                and s[0] not in s[2:]                     # (2)
                                and o[0] == 0                             # (3)
                                and o[1] in [0, 1]                        # (4)
                        ):
                            # this_Omega_p[i, m] = 1                              # possible observation
                            rows[idx] = i
                            cols[idx] = m
                            idx += 1

                        # CONDITION "STEP B":            CORRESP MODEL VARIABLE:
                        # ------------------------------------------------------
                        # if new position...                               s[0]
                        # ...(1) IS NOT treasure location                  s[1]
                        # ...(2) IS hiding spot,                           s[2:]
                        # all observation, for which...
                        # ...(3) tr_flag == 0,                             o[0]
                        # ...(4) and node color in [0, 1](black or blue),  o[1]

                        if (                                                    # CONDTION "STEP B"
                                s[0] != s[1]                              # (1)
                                and s[0] in s[2:]                         # (2)
                                and o[0] == 0                             # (3)
                                and o[1] in [0, 2]                        # (4)
                        ):
                            # this_Omega_p[i, m] = 1                              # possible observation
                            rows[idx] = i
                            cols[idx] = m
                            idx += 1

                        # CONDITION "STEP C":            CORRESP MODEL VARIABLE:
                        # ------------------------------------------------------
                        # if new position...                               s[0]
                        # ...(1) IS treasure location                      s[1]
                        # ...(2) IS hiding spot,                           s[2:]
                        # all observation, for which...
                        # ...(3) tr_flag == 1,                             o[0]
                        # ...(4) and node color in [0, 1](black or blue),  o[1]

                        if (                                                    # CONDTION "STEP C"
                                s[0] == s[1]                              # (1)
                                and s[0] in s[2:]                         # (2)
                                and o[0] == 1                             # (3)
                                and o[1] in [0, 2]                        # (4)
                        ):
                            # this_Omega_p[i, m] = 1                              # possible observation
                            rows[idx] = i
                            cols[idx] = m
                            idx += 1

                end = time.time()
                print(f"Finished one state iteration, for , s = {s} i: {i}, idx: {idx}, "
                      f"time needed: {humanreadable_time(end-start)}")

            # Omega[p] = this_Omega_p.tocsc()                                   # transform lil_matrix to csc_matrix

            # Create action-dependent Pmega[p] as sparse matrix
            Omega[p] = sp.csc_matrix(
                ([1] * n_nonzeros,                                              # data values
                 (rows, cols)),                                                 # row and column indices, for data values
                shape=(n_s, n_o),                                               # shape of matrix
                dtype=np.int8                                                   # datatype
            )

            # Save Omega[p] to disk
            paths.save_arrays(
                sparse=True,
                file_name=Omega_matrix_name,
                array=Omega[p],
            )  # TODO: robust coden, speichert noch alle hitherto erstellten Phi's auf einmal

        # Load Omega[p] from disk, if existing
        else:
            with open(os.path.join(paths.components, f"{Omega_matrix_name}.npz"), "rb") as file:
                Omega[p] = sp.load_npz(file)

    # Plot Omega[p]s, only if grid is of small dimension d = 2
    if d == 2:
        plot_color_map(
            paths=paths,
            Omega_drill=Omega[0].todense(),
            Omega_step=Omega[1].todense(),
        )

    return Omega
