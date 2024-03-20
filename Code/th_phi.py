import os
import numpy as np                                                              # numpy
from th_imshow import plot_color_map
import scipy.sparse as sp
from th_helper import humanreadable_time
import time


def th_phi(S, A, theta, paths):
    """"
    This function evaluates the action-dependent and state-conditional
    observation probability distribution of a Bayesian agent for the treasure
    hunt task.
    If Phi.npy file exists, Phi is loaded from disk, otherwise
    evaluated and saved to disk.

    Inputs
        theta     : task parameter structure with required fields
            .n_n  : number of nodes
            .n_h  : number of hiding spots
            .n_s  : state space cardinality
            .n_a  : action space cardinality
        S         : n_s x 1 + n_h state set array
        A         : n_a x 0 action set array

    Outputs
        Phi      : n_s x n_s x 5 transition probability matrices array

    Saves to disk (if not existing)
        Phi.npy  : n_s x n_s x 5 transition probability matrices array

    Authors - Belinda Fleischmann, Dirk Ostwald
    """
    # Task parameters and set cardinalities
    n_n     = theta.n_n                                                         # number of nodes
    n_h     = theta.n_h                                                         # number of hiding spots
    d       = theta.d                                                           # dimension of the square grid world
    n_s     = theta.n_s                                                         # state space cardinality (number of states)
    n_s3    = theta.n_s3                                                        # number of unique hiding spot combination possibilities
    n_a     = theta.n_a                                                         # action space cardinality (number of actions)

    # Initialize dictionary of state-state transition probability matrices
    Phi = {}
    for p in range(n_a):                                                        # iterate action indices
        Phi[p] = sp.csc_matrix(
            (n_s, n_s),                                                         # shape of Phi[:, :, a]
            dtype=np.int8                                                       # smallest possible datatype: int8
        )

    # Define list of labels for the action-specific Phi matrices; used for saving or loading matrices from disk
    matrix_names = [
        "Phi_drill",
        "Phi_minus_dim",
        "Phi_plus_one",
        "Phi_plus_dim",
        "Phi_minus_one"
    ]

    # --------------------------------------------------------------------------
    # Evaluation of Phi values
    # --------------------------------------------------------------------------
    # To minimize computation time, the following describes the workaround that
    # avoids looping through all n_s states in both nested loops.
    # Since phi values only depends on s[0] (current position),it suffices to
    # evaluate if s_1_t + a_t == s_1_tt. The remaining state components (s[1:])
    # remain constant over trials, regardless of a_t.
    # Thus, all tuples (s_t, s_tt), for which s1_t == s1_tt, evaluate to 1, if
    # s_2:_t == s_2:_tt, and zero, else. This results in n_n s1-specific
    # identity matrices within Phi, whith the size n_ident, which corresesponds
    # to the the number of state values (s) per value s_1.
    # For each possible value s_1 = i for the current position, there are
    # n_ident = n_s3 * n_h state values s, in which s_1 = i.

    # Knowing this, it suffices to only identify the "starting coordinates" of
    # those identiy matrices, and then to (fastforward) save respective row and
    # col indices. More specifically for each possible value s1_t, which
    # extends over n_ident rows in Phi, we need to identify the first column j,
    # in which s1_tt == s1_t + a. That is then the starting column of the
    # s1-specific identity matrix. from there respective row and column vectors
    # are sequences starting with respective row and column indices i and j,
    # and ending at i + n_ident and j + ident, respectively

    # e.g. if n_n = 4, n_h = 2, and n_s = 48, there are 4 identity matrices
    # in Phi and n_ident = 12, and i=0, j=12 are the starting coordinates of an
    # identity matrix, respective row and col indices would be
    # np.arange(0, 12) = np.arange(i, i + n_ident) for rows, and
    # np.arange(12, 24) = np.arange(j, j + n_ident) for cols.

    # ----------------
    # EXAMPLE values
    # ----------------
    # in lines            0 : (n_s3 * n_h)    , S[0] == 1
    # in lines (n_s3 * n_h) : (2 * n_s3 * n_h), S[0] == 2
    # Test with:
    # np.all(S[0: n_s3 * n_h, :][:, 0] == 1)
    # np.all(S[n_s3 * n_h : 2 * n_s3 * n_h, :][:, 0] == 2)
    # --------------------------------------------------------------------------

    n_nonzeros = n_n * n_h * n_s3                                               # number of nonzero values in each Phi[p], same as n_s

    # Create row index iterable for s1-sepcific identiy matrices
    n_ident = n_s // n_n                                                        # size of one identity matrix, same as n_s3 * n_h step size for iterable
    iterable_I_matrix = range(0, n_s, n_ident)                                  # iterable to compute Phi identity matrix wise

    # -----------------------------------------------------------------------------------------------------
    # Compute or load action-dependent and state-conditional observation probability distribution matrices
    # -----------------------------------------------------------------------------------------------------
    for p, a in enumerate(A):                                                   # a_t iterations

        Phi_matrix_name = matrix_names[p]                                       # Get action-specific Matrix label string for path variable

        # Compute Phi[p] if not existing on disk
        if not os.path.exists(os.path.join(paths.components, f"{Phi_matrix_name}.npz")):

            print(f"Starting evaluation of Phi[{p}]..., matrix_label: {Phi_matrix_name}")
            # Initialize arrays to store row and col indices, that indicate value 1 entries in Phi[p]; needed to create sparse matrices
            rows = np.full((n_nonzeros), np.nan)                                # array to store row indices
            cols = np.full((n_nonzeros), np.nan)                                # array to store col indices

            # Iterate s1-specific identiy matrices; i is the row index of where one identity matrix "starts", i.e. smalles row index of respective I-matrix
            for i in iterable_I_matrix:

                start = time.time()

                for j in range(n_s):                                            # s_t+1 iterations

                    a_aug     = np.hstack([a, np.zeros(S.shape[1] - 1)])        # a augmented for addition to state vector
                    s_t     = S[i, :]                                           # s_t
                    s_tt    = S[j, :]                                           # s_{t+1}
                    s1_t    = s_t[0]                                            # first element of s_t, representing current position in t

                    # If action is valid (movement within grid, no boarder crossing)
                    if (
                        # a does not move the agent beyond the top or bottom border
                        (1 <= (s1_t + a) <= n_n)                                # new position is a valid node number (i.e \in [1, n_n])
                        # a does not move the agent beyond the left boarder
                        and not (
                            (a == -1)                                           # move to the left
                            and (((s1_t - 1) % d) == 0))                        # while standing on most left column of the grid
                        # a does not move the agent beyond right boarder
                        and not (
                            (a == 1)                                            # move to the right
                            and ((s1_t % d) == 0))                              # while standing on most right column of the grid
                    ):

                        # Set Phi-entry that represents correct state transition to 1
                        if np.array_equal(s_t + a_aug, s_tt):                 # TODO: p(s_{t+1} = \tilde{s} |s_{t} = s) = 1 for \tilde{s} = s + a_augm, 0 else

                            rows[i: (i + n_ident)] = np.arange(i, i + n_ident)
                            cols[i: (i + n_ident)] = np.arange(j, j + n_ident)
                            break                                               # move to next i iteration (since per s_1_t, only ONE possible s_1_tt)

                    # If action is invalid (movement that crosses grid boarders)
                    else:
                        # NOTE:
                        # ------------------------------------------------------
                        # According to task rules, invalid actions are not
                        # recorderd (counted). Instead, participants can repeat
                        # the action decision. Thus, the following represents
                        # the agent's belief to stay on its current position,
                        # i.e. first state component s^1_tt = s^1_t + a_t, while
                        # all other state components remain the same, as well.
                        # In other words, ALL state components remain the same
                        # ------------------------------------------------------

                        if np.array_equal(s_t, s_tt):                           # check if all state components remain same
                            rows[i: i + n_ident] = np.arange(i, i + n_ident)
                            cols[i: i + n_ident] = np.arange(j, j + n_ident)
                            break                                               # move to next i iteration (since per s_1_t, only ONE possible s_1_tt)

                end = time.time()

                print(f"Finished one iteration, for current position, s[1] = {s1_t} i: {i}, j: {j}, "
                      f"time needed: {humanreadable_time(end-start)}")

            # Create action-dependent Phi[p] as sparse matrix
            Phi[p] = sp.csc_matrix(
                ([1] * n_nonzeros,                                              # data values
                 (rows, cols)),                                                 # row and column indices, for data values
                shape=(n_s, n_s),                                               # shape of matrix
                dtype=np.int8                                                   # datatype
            )

            # Save Phi[p] to disk
            paths.save_arrays(
                sparse=True,
                file_name=Phi_matrix_name,
                array=Phi[p]
            )  # TODO: robust coden, speichert noch alle hitherto erstellten Phi's auf einmal

        # Load Phi[p] from disk, if existing
        else:
            with open(os.path.join(paths.components, f"{Phi_matrix_name}.npz"), "rb") as file:
                Phi[p] = sp.load_npz(file)

    # Plot Phi[p]s, only if grid is of small dimension d = 2
    if d == 2:
        plot_color_map(                                                     # plot action specific Phi matrices
            paths=paths,
            Phi_drill=Phi[0].todense(),
            Phi_minus_dim=Phi[1].todense(),
            Phi_plus_one=Phi[2].todense(),
            Phi_plus_dim=Phi[3].todense(),
            Phi_minus_one=Phi[4].todense()
        )

    return Phi
