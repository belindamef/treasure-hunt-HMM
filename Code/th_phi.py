import os
import numpy as np                                                              # numpy
from th_imshow import plot_color_map


def th_phi(S, A, theta):
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
    n_h     = theta.n_h                                                         # number of treasure hiding spots
    d       = theta.d                                                           # dimension of the square grid world
    n_s     = theta.n_s                                                         # state space cardinality (number of states)
    n_a     = theta.n_a                                                         # action space cardinality (number of actions)

    # Initialize state-state transition probability matrices array
    Phi     = np.full([n_s, n_s, n_a], 0, dtype=int)                            # action-dependent state-state transition probability matrices array initialization

    # Compute Phi if not existing on disk
    if not os.path.exists("Components/Phi.npy"):

        for k in range(n_a):                                                    # a_t   iterations

            for i in range(n_s):                                                # s_t   iterations

                for j in range(n_s):                                            # s_t+1 iterations

                    a_t = A[k]                                                  # a_t action scalar
                    a_t_aug     = np.hstack([a_t, np.zeros(S.shape[1] - 1)])    # a_t augmented for addition to state vector
                    s_t     = S[i, :]                                           # s_t
                    s_tt    = S[j, :]                                           # s_{t+1}
                    s1_t    = s_t[0]                                            # first element of s_t, representing current position in t

                    # If action is valid (movement within grid, and boarder crossing)
                    if (
                        # a move that does not move the agend beyond the top or bottom border
                        (1 <= (s1_t + a_t) <= n_n)                              # new position is a valid node number (i.e \in [1, n_n])
                        # not a move beyond the left boarder
                        and not ((a_t == -1)                                    # not movement to the left
                                 and (((s1_t - 1) % d) == 0))                   # while standing on most left column of the grid
                        # not a move beyond right boarder
                        and not ((a_t == 1)                                     # not a movement to the right
                                 and ((s1_t % d) == 0))                         # while standing on most right column of the grid
                    ):

                        # Set Phi-entry that represents correct state transition to 1
                        if np.array_equal(s_t + a_t_aug, s_tt):                 # TODO: p(s_{t+1} = \tilde{s} |s_{t} = s) = 1 for \tilde{s} = s + a_augm, 0 else
                            Phi[i, j, k] = 1

                    # If action is invalid (movement that crosses grid boarders)
                    else:
                        # According to task rules, invalid actions are not recorderd or counted
                        # Instead, participants can repeat the action decision
                        # Thus, the following represents the agent's belief to stay on its current position,
                        # i.e. first state component s^1_tt = s^1_t + a_t, while all other state components remain the same, as well
                        # in other words, ALL state components remain the same
                        if np.array_equal(s_t, s_tt):
                            Phi[i, j, k] = 1                                    # TODO: p(s_{t+1} = \tilde{s} |s_{t} = s) = 1 for \tilde{s} = s, 0 else

        np.save("Components/Phi", Phi)                                          # save to disc

        plot_color_map(                                                         # plot action specific Phi matrices
            n_nodes=n_n,
            n_hides=n_h,
            Phi_drill=Phi[:, :, 0],
            Phi_minus_dim=Phi[:, :, 1],
            Phi_plus_one=Phi[:, :, 2],
            Phi_plus_dim=Phi[:, :, 3],
            Phi_minus_one=Phi[:, :, 4]
        )

    # Else load Phi from disk
    else:
        Phi = np.load("Components/Phi.npy")

    return Phi
