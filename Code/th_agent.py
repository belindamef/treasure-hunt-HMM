import numpy as np                                                              # numpy


class th_agent:
    def __init__(self, a_init):
        """
        This function encodes the instantiation method of the treasure hunt 
        agent class.

        Inputs
            a_init     : agent initialization parameter structure with fields
                .task  : dimensionality of the square grid world
        Output
            initialized agent object

        Authors - Belinda Fleischmann, Dirk Ostwald
        """
        # structural components
        self.task   = a_init.task                                               # task information

        # dynamic components
        self.c      = np.nan                                                    # current round
        self.t      = np.nan                                                    # current trial
        self.b      = np.nan                                                    # current belief state
        self.d      = np.nan                                                    # current decision

    def delta(self):
        """
        This function implements the agent's decision function delta.

        Input
            self    : agent object

        Output
            self    : agent object with updated attribute
                d   : decision
        """
        self.d  = 0
