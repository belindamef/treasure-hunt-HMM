import numpy as np                                                              # numpy
import scipy.stats as rv                                                        # random variable module


class th_task:
    def __init__(self, t_init):
        """
        This function encodes the instantiation method of the treasure hunt
        task class.

        Inputs
            t_init  : task initialization parameter structure with fields
                .theta  : task parameters
                .S      : state set
                .O      : observation set
                .A      : action set
                .Phi    : action-dependent state-state transition probability

        Authors - Belinda Fleischmann, Dirk Ostwald
        """
        # structural components
        self.theta  = t_init.theta                                              # task parameters
        self.S      = t_init.S                                                  # state set
        self.O      = t_init.O                                                  # observation set
        self.A      = t_init.A                                                  # action set
        self.Phi    = t_init.Phi                                                # action-dependent state-state transition probability

        # dynamic components
        self.r      = np.nan                                                    # current round
        self.t      = np.nan                                                    # current trial
        self.s      = np.nan                                                    # task state
        self.o      = np.nan                                                    # task observation
        self.a      = np.nan                                                    # agent action
        self.i_s    = np.nan                                                    # 

    def start_game(self):
        """
        This function determines a game's starting state, including the agent's
        starting position, the treasure position and the hiding spot positions.
        It keeps on sampling until the starting positon is not the treasure
        location.

        Inputs
                self    : task object

        Outputs
                self    : task object with updated attributes
                    .s_i: task state index
                    .s  : task state
        """

        while True:
            self.i_s        = rv.randint.rvs(0, self.theta.n_s)                 # uniform random state index
            self.s          = self.S[self.i_s, :]                               # state value
            # check, if current position at beginning of a game is treasure location
            if self.s[0] != self.s[1]:                                          # current positon == treasure location?
                break

    def f(self, i_a):
        """"
        This function evaluates the task's state-state transition function.

        Inputs
            self    : task object
            a       : agent action

        Outputs
            self      : task object with updated attribute
                .i_s  : task state index
                .s    : task state
        """
        i_s_tt = np.argmax(                                                     # index of s_{t+1}
            rv.multinomial.rvs(1, self.Phi[i_a][self.i_s, :]) != 0)
        s_tt   = self.S[self.s_i, :]                                            # s_{t+1}
        # if i_a != 0 and np.sum(self.Phi[i_a][self.s_i, :]) == 0:
        if i_a != 0 and s_tt == self.s:                                         # if after step action the new state is the same as before
            print("Invalid action")
            # TODO: dann stop? oder wie signalisieren, dass das trial wiederholt wird? 
        else:
            self.i_s = i_s_tt
            self.s   = s_tt
