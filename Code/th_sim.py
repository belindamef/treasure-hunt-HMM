"""
This Python script simulates the observation of a task-agent interaction on
a single game of the treasure hunt task.

Authors - Belinda Fleischmann, Dirk Ostwald
"""
# dependencies
import numpy as np                                                              # numpy
import os                                                                       # operating system interface
from th_structure import th_structure                                           # structures
from th_paths import th_paths                                                   # path variables
from th_cards import th_cards                                                   # task sets' cardinalities
from th_sets import th_sets                                                     # task/agent model sets generator
from th_phi import th_phi                                                       # action-dependent state-state transition probability matrices
from th_omega import th_omega                                                   # action-dependent state conditional observation probability matrices

# directory management
this_task_config_label = "test"
paths = th_structure()

# task parameters
theta           = th_structure()                                                # simulation structure initialization
theta.d         = 2                                                             # dimension of the square grid world
theta.n_n       = theta.d ** 2                                                  # number of grid world cells/nodes
theta.n_h       = 2                                                             # number of treasure hiding spots
theta.d_s       = 2 + theta.n_h                                                 # state vector dimension (agent location, treasure location, hiding spot locations)
theta.n_r       = 1                                                             # number of rounds per game
theta.n_t       = 12                                                            # maximal number of actions per round
theta           = th_cards(theta)                                               # task sets' cardinalities

# define path to model components

paths = th_paths(theta)

# task sets
theta           = th_sets(theta, paths)                                         # task/agent model state, observation, decision, and action set creation
S               = np.load(os.path.join(paths.components, "S.npy"))              # state set
O               = np.load(os.path.join(paths.components, "O.npy"))                          # observation set
A               = np.load(os.path.join(paths.components, "A.npy"))                          # action set

# stochastic matrices
Phi             = th_phi(S, A, theta, paths)                                    # action-dependent state-state transition probability matrices
Omega           = th_omega(S, O, theta, paths)                                  # action-dependent state conditional observation probability matrices
