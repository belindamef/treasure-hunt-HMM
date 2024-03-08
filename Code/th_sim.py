"""
This Python script simulates the observation of a task-agent interaction on
a single game of the treasure hunt task.

Authors - Belinda Fleischmann, Dirk Ostwald
"""
# dependencies
import numpy as np                                                              # numpy
from th_structure import th_structure                                           # structures
from th_cards import th_cards                                                   # task sets' cardinalities
from th_sets import th_sets                                                     # task/agent model sets generator
from th_phi import th_phi                                                       # action-dependent state-state transition probability matrices
from th_omega import th_omega                                                   # action-dependent state conditional observation probability matrices
from th_task import th_task                                                     # task object
from th_sim_game import th_sim_game                                             # game simulation routine
from th_vis_game import th_vis_game                                             # game visualization routine

# task parameters
theta           = th_structure()                                                # simulation structure initialization
theta.d         = 3                                                             # dimension of the square grid world
theta.n_n       = theta.d ** 2                                                  # number of grid world cells/nodes
theta.n_h       = 1                                                             # number of treasure hiding spots
theta.d_s       = 2 + theta.n_h                                                 # state vector dimension (agent location, treasure location, hiding spot locations)
theta.n_r       = 1                                                             # number of rounds per game
theta.n_a       = 12                                                            # maximal number of actions per round
theta           = th_cards(theta)                                               # task sets' cardinalities
