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
from th_sim_game import th_sim_game                                             # game simulation routine


# directory management
this_task_config_label = "test"
paths = th_structure()

# task parameters
theta           = th_structure()                                                # simulation structure initialization
theta.d         = 5                                                             # dimension of the square grid world
theta.n_n       = theta.d ** 2                                                  # number of grid world cells/nodes
theta.n_h       = 6                                                             # number of treasure hiding spots
theta.d_s       = 2 + theta.n_h                                                 # state vector dimension (agent location, treasure location, hiding spot locations)
theta.n_c       = 1                                                             # number of rounds per game
theta.n_t       = 12                                                            # maximal number of actions per round
theta           = th_cards(theta)                                               # task sets' cardinalities

# define path to model components

paths = th_paths(theta)

# task sets
theta           = th_sets(theta, paths)                                         # task/agent model state, observation, decision, and action set creation
S               = np.load(os.path.join(paths.components, "S.npy"))              # state set
O               = np.load(os.path.join(paths.components, "O.npy"))              # observation set
A               = np.load(os.path.join(paths.components, "A.npy"))              # action set

# stochastic matrices
Phi             = th_phi(S, A, theta, paths)                                    # action-dependent state-state transition probability matrices
Omega           = th_omega(S, O, theta, paths)                                  # action-dependent state conditional observation probability matrices

# task initialization structure
t_init          = th_structure()                                                # task initialization structure
t_init.theta    = theta                                                         # task parameters
t_init.S        = S                                                             # state set
t_init.O        = O                                                             # observation set
t_init.A        = A                                                             # action set
t_init.Phi      = Phi                                                           # action-dependent state-state transition probability matrix

# agent initialization structure
a_init          = th_structure()                                                # task initialization structure
a_init.Omega    = Omega                                                         # action-dependent state conditional observation probability matrices

# simulation
sim             = th_structure()                                                # game simulation structure initialization
sim.p           = 1                                                             # participant index
sim.g           = 1                                                             # game index
sim.mode        = "simulation"                                                  # simulation mode
sim.theta       = theta                                                         # simulation parameters
sim.t_init      = t_init                                                        # task initialization structure
sim.a_init      = a_init                                                        # agent initialization structure
sim             = th_sim_game(sim)                                              # simulate one treasure hunt game
