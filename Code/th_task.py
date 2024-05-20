import numpy as np                                                              # numpy
import scipy.stats as rv


class th_task:
    def __init__(self, t_init):
        """This function encodes the instantiation method of the treasure hunt
        task class.

        Inputs
            t_init      (obj) : task initialization parameter structure with fields
                .theta  (obj) : task parameters
                .S      (arr) : n_s x (2 + n_h) array of state values
                .O      (arr) : n_n x 2 array of observation values
                .A      (arr) : 5 x 1 array of action values
                .Phi    (arr) : n_s x n_s x 5 matrices array of state transition probabilities

        Authors - Belinda Fleischmann, Dirk Ostwald
        """
        # Structural components
        self.theta       = t_init.theta                                         # task parameters
        self.S           = t_init.S                                             # state set
        self.O           = t_init.O                                             # observation set
        self.A           = t_init.A                                             # action set
        self.Phi         = t_init.Phi                                           # action-dependent state-state transition probability
        self.A_giv_s1    = np.nan                                               # state-dependent action set

        # Dynamic components
        self.c           = np.nan                                               # current round
        self.t           = np.nan                                               # current trial
        self.s           = np.nan                                               # task state  # TODO 8 values??
        self.i_s         = np.nan                                               # state index
        self.o           = np.full(2, np.nan, dtype=int)                        # task observation
        self.a           = np.nan                                               # agent action
        self.r           = np.nan                                               # reward
        self.node_colors = np.full(self.theta.n_n, 0, dtype=np.int8)            # current node colors, initialized as zeros (all black)


    def start_game(self):
        """
        This function determines a game's starting state, including the agent's
        starting position, the treasure position and the hiding spot positions.
        It keeps on sampling until the starting positon is not the treasure
        location.

        Inputs
                self   (obj) : task object

        Outputs
                self     (obj) : task object with updated attributes
                    .s_i (int) : task state index
                    .s   (arr) : 1 x (n_h + 2) array of current task state
        """

        while True:
            self.i_s        = rv.randint.rvs(0, self.theta.n_s)                 # uniform random state index
            self.s          = self.S[self.i_s, :]                               # state value
            # check, if start position at beginning of a game is treasure loc
            if self.s[0] != self.s[1]:                                          # current positon == treasure location?
                break

    def f(self, a):
        """This function evaluates the task's state-state transition function.

        Inputs
            self     (obj) : task object
                .i_s (int) : state index
                .s   (arr) : 1 x (n_h + 2) array of current task state (to be updated)
                .S   (arr) : n_s x (2 + n_h) array of state values
                .Phi (arr) : n_s x n_s x 5 matrices array of state transition probabilities
            a        (int) : action in trial t

        Outputs
            self     (obj) : task object with updated attributes
                .i_s (int) : task state index
                .s   (arr) : 1 x (n_h + 2) array of current task state (updated)
        """
        a_i = int(np.where(self.A == a)[0])                                     # action index

        Phi_a_s_t = self.Phi[a_i][self.i_s, :].toarray()[0]                     # a-dep. Phi vector giv current s_t
        i_s_tt = np.argmax(                                                     # index of s_{t+1}
            rv.multinomial.rvs(
                1, Phi_a_s_t
            ) != 0
        )
        s_tt = self.S[i_s_tt, :]                                                # s_{t+1}

        if a != 0 and np.all(s_tt == self.s):                                   # after step action the new state is the same as before
            print("Invalid action")
            # Das sollte gar nicht erst passieren können, da agent nur von A_giv_s1 wählt
        else:
            self.i_s = i_s_tt                                                   # s_{t} index
            self.s   = s_tt                                                     # s_{t}

    def update_node_colors(self):
        """This function updated the node colors after drill action.

        Inputs
            self     (obj) : task object
                .s   (arr) : 1 x (n_h + 2) array of current task state (to be updated)

        Outputs
            self     (obj) : task object with updated attributes
                .s   (arr) : 1 x (n_h + 2) array of current task state (updated)
        """
        s1 = self.s[0]                                                          # current position s^1
        i_s1 = s1 - 1                                                           # current position's node index
        s3 = self.s[2:]                                                         # hiding spots

        if s1 in s3:                                                            # if current (drilled-on) position is hiding spot
            self.node_colors[i_s1] = 2                                          # unveal hiding spot, i.e. set node color to blue
        else:                                                                   # if current (drilled-on) position is not hiding spot
            self.node_colors[i_s1] = 1                                          # unveal non-hiding spot, i.e. set node color to gray

    def g(self):
        """This function evaluates the task's observation function.

        Inputs
            self             (obj) : task object
                .r           (int) : reward
                .node_colors (arr) : TODO
        Outputs
            self             (obj) : task object with updated attributes
                .o           (arr) : 1 x 2 array of observation
        """
        # TODO: hier weiter
        a_i = int(np.where(self.A == a)[0])                                     # action index

        Phi_a_s_t = self.Phi[a_i][self.i_s, :].toarray()[0]                     # Phi vector giv current a and s_t
        i_s_tt = np.argmax(                                                     # index of s_{t+1}
            rv.multinomial.rvs(
                1, Phi_a_s_t
            ) != 0
        )
        s_tt = self.S[i_s_tt, :]                                                # s_{t+1}

        if a != 0 and np.all(s_tt == self.s):                                   # after step action the new state is the same as before
            print("Invalid action")
            # Das sollte gar nicht erst passieren können, da agent nur von A_giv_s1 wählt
        else:
            self.i_s = i_s_tt                                                   # s_{t} index
            self.s   = s_tt                                                     # s_{t}

        # Update treasure flag o[0]
        if self.r == 0:                                                         # no treasure found
            self.o[0] = 0                                                       # treasure flag = 0
        else:                                                                   # treausure found
            self.o[0] = 1                                                       # treasure flag = 1

        # Update color observation on current postion o[1]
        self.o[1] = 0 #self.node_colors[self.s[0]]                              # o^2 corresponds to color on current position

        # TODO: hier weiter
        # * o_t soll aus Omega gesampled werden (ist natürlich deterministisch)

    def identify_A_giv_s1(self):
        """This function evaluates the state dependent set of actions

        Inputs
            self           (obj) : task object
                .A         (arr) : 5 x 1 array of action values
                .s         (arr) : 1 x (n_h + 2) array of current task state (updated)
                .theta.d   (int) : dimensionality of the square grid world
                .theta.n_n (int) : number of nodes

        Outputs
            self           (obj) : task object with updated attributes
                .A_giv_s1  (arr) : n_A_s1 x 1 array of action values TODO: varying n_A_s1
        """
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
