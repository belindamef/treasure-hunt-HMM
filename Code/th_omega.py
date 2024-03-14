import numpy as np
import os
from th_imshow import plot_color_map


def th_omega(S, O, theta):
    """This function evaluates the action-dependent and state-conditional
    observation probability distribution of a Bayesian agent for the treasure
    hunt task.

    Inputs:
        S       : n_s x 1 + n_h array
        O       : (TODO)
        theta   : task parameter structure with required fields

    Outputs:
        Omega   : n_s x n_o x 2 observation probability distribution

    """
    # Task parameters and set cardinalities
    n_n     = theta.n_n                                                         # number of nodes
    n_h     = theta.n_h                                                         # number of treasure hiding spots
    n_s     = theta.n_s                                                         # state space cardinality (number of states)
    n_o     = theta.n_o                                                         # observation space cardinality (number of observations)
    n_a     = 2                                                                 # compressed action space cardinality (drill/step)

    # Action set
    A = [0, 1]                                                                  # compressed action space (drill/step)

    # Initialize observation probability distribution array
    Omega   = np.full([n_s, n_o, 2], 0, dtype=int)

    # Compute Omega if not existing on disk
    if not os.path.exists("Components/Omega.npy"):

        for p in range(n_a):                                                    # action iterations

            for i in range(n_s):                                                # state iterations

                for m in range(n_o):                                            # observation iterations
                    # TODO: FRAGE: a, s, o oder a_t, s_t, o_t
                    # TODO: FRAGE: warum nicht direkt for a in A, s in S und o in O

                    s = S[i, :]
                    o = O[m, :]
                    a = A[p]

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
                            Omega[i, m, p] = 1                                  # possible observation

                        # CONDITION "DRILL B":           CORRESP MODEL VARIABLE:
                        # ---------------------------------------------------------
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
                            Omega[i, m, p] = 1                                  # possible observation

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
                            Omega[i, m, p] = 1                                  # possible observation

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
                            Omega[i, m, p] = 1                                  # possible observation

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
                            Omega[i, m, p] = 1                                  # possible observation

        np.save("Components/Omega", Omega)                                      # save to disc

        plot_color_map(
            n_nodes=n_n,
            n_hides=n_h,
            Omega_drill=Omega[:, :, 0],
            Omega_step=Omega[:, :, 1]
        )

    # Else load Omega from disk
    else:
        Omega = np.load("Components/Omega.npy")
    return Omega
