import numpy as np                                                              # numpy
import scipy.stats as rv


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
        self.theta    = t_init.theta                                            # task parameters
        self.S        = t_init.S                                                # state set
        self.O        = t_init.O                                                # observation set
        self.A        = t_init.A                                                # action set
        self.Phi      = t_init.Phi                                              # action-dependent state-state transition probability
        self.A_giv_s1 = np.nan                                                  # state-dependent action set

        # dynamic components
        self.c           = np.nan                                               # current round
        self.t           = np.nan                                               # current trial
        self.s           = np.nan                                               # task state  # TODO 8 values??
        self.i_s         = np.nan                                               # state index
        self.o           = np.nan                                               # task observation
        self.a           = np.nan                                               # agent action
        self.r           = np.nan                                               # reward
        self.node_colors = np.nan                                               # current node colors

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
            # check, if start position at beginning of a game is treasure loc
            if self.s[0] != self.s[1]:                                          # current positon == treasure location?
                break

    def f(self, a):
        """
        This function evaluates the task's state-state transition function.

        Inputs
            self    : task object
            a       : agent action

        Outputs
            self      : task object with updated attribute
                .i_s  : task state index
                .s    : task state
        """
        a_i = int(np.where(self.A == a)[0])

        Phi_a_s_t = self.Phi[a_i][self.i_s, :].toarray()[0]                     # a-dep. Phi vector giv current s_t
        i_s_tt = np.argmax(                                                     # index of s_{t+1}
            rv.multinomial.rvs(
                1, Phi_a_s_t
            ) != 0
        )
        s_tt   = self.S[i_s_tt, :]                                              # s_{t+1}

        if a != 0 and np.all(s_tt == self.s):                                   # after step action the new state is the same as before
            print("Invalid action")
            # Das sollte gar nicht erst passieren können, da agent nur von A_giv_s1 wählt
        else:
            self.i_s = i_s_tt
            self.s   = s_tt

    def g(self):
        """
        This function evaluates the task's observation

        Inputs
            self      : task object
                .r_t  : rewart
                ."""

    def identify_A_giv_s1(self):
        """Identify state s1 dependent action set"""
        self.A_giv_s1 = self.A

        for action in self.A:
            new_s1 = action + self.s[0]
            # Remove forbidden steps (walk outside border)
            if (not (1 <= new_s1 <= self.theta.n_n)
                    or (((self.s[0] - 1)
                         % self.theta.d == 0)
                        and action == -1)
                    or (((self.s[0])
                         % self.theta.d == 0)
                        and action == 1)):

                self.A_giv_s1 = self.A_giv_s1[self.A_giv_s1 != action]
