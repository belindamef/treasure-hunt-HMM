import numpy as np                                                              # numpy


class th_agent:
    def __init__(self, a_init):
        """
        This function encodes the instantiation method of the treasure hunt 
        agent class.

        Inputs
            a_init     (obj) : agent initialization parameter structure with fields
                .task  (obj) : task object

        Authors - Belinda Fleischmann, Dirk Ostwald
        """
        # structural components
        self.task   = a_init.task                                               # task information

        # dynamic components
        self.c      = np.nan                                                    # current round
        self.t      = np.nan                                                    # current trial
        self.b      = np.nan                                                    # current belief state
        self.v      = np.nan                                                    # current action valences
        self.d      = np.nan                                                    # current decision

    def delta(self):
        """
        This function implements the agent's decision function delta.

        Input
            self   (obj) : agent object

        Output
            self   (obj) : agent object with updated attribute
                .d (int) : decision
        """
        self.d = np.random.choice(self.task.A_giv_s1, 1)
        return self.d
