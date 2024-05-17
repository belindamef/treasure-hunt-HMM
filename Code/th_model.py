import copy as cp
import numpy as np


class th_model:
    def __init__(self, m_init):
        """
        This function encodes the instantiation method of the treasure hunt
        behavioral model class.

        Inputs
            m_init     (obj) : model initialization parameter structure with fields
                .agent (obj) : agent object
                .task  (obj) : task object
                .tau   (obj) : post-decision noise parameter

        Authors - Belinda Fleischmann, Dirk Ostwald
        """
        self.agent = m_init.agent                                               # agent object
        self.task  = m_init.task                                                # task object
        self.tau   = m_init.theta.tau                                           # post decision noise parameter
        self.a     = np.nan                                                     # action in trial t

    def eval_p_a_giv_history_o_and_a(self):
        """This function evaluates the conditional probability distribution of
        of actions given the history of actions and observations and tau

        Input
            self              (obj) : model object
                .tau        (float) : post decision parameter
                .agent.v      (arr) : (1 X n_A_s1?) array current action valences TODO: varying n_A_s1

        Output
            self              (obj) : model object with updated attribute
                .p_a_giv_hist (arr) : (1 X n_A_s1?) array representing the probability distribution of action given the history of actions and observations
        """
        return np.exp((1 / self.tau) * self.agent.v) / sum(
            np.exp((1 / self.tau) * self.agent.v))

    def return_action(self):
        """This function returns an action given the agent's decision or
        valence functions

        Inputs
            self               (obj) : model object
                .tau         (float) : post decision parameter

                .agent.d       (int) : agent decision in trial t
                .task.A_giv_s1 (arr) : (1 X n_A_s1?) array of state-dependent action set TODO: varying n_A_s1

        Outputs
            self               (obj) : model object with updated attribute
                .a             (int) : action in trial t
        """
        # Direct transmission of agent decision to action
        if (np.isnan(self.tau) or self.tau == 0):
            self.a = cp.deepcopy(self.agent.d)

        # Sample action with softmax operation
        else:
            p_a_giv_h    = self.eval_p_a_giv_tau()                              # probabillity distribution of action given history
            rng          = np.random.default_rng()                              # randon numer generator
            rvs          = rng.multinomial(1, p_a_giv_h)                        # sample one random variate from multinomial distribution
            action_index = rvs.argmax()                                         # get corresponding action index
            self.a       = self.task.A_giv_s1[action_index]                     # get correspoinding action value

        return self.a
