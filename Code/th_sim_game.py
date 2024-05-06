import numpy as np                                                              # NumPy
import pandas as pd                                                             # Pandas
from th_task import th_task                                                     # task model module
from th_agent import th_agent                                                   # agent model module
import logging

def th_sim_game(sim):
    """
    This function simulates the experimental observation of an interaction
    between the treasure hunt task and an agent model on a single game.
    It works in two fundamental modes:

    - In full simulation mode, relevant variables (states, observations, actions)
      are sampled  according to the respective model probability distributions.
    - In partial simulation mode, relevant variables (states, rewards,
      actions) are not sampled, but read from the experimental data set.

    Inputs:
        sim  : structure with fields
            .p      : participant index
            .mode   : simulation mode
         .g      : game index
               .theta  : simulation parameters
            .t_init : task initialization structure
            .a_init : agent initialization structure
            .m_init : model initialization structure


    Outputs
        sim  : structure equivalent with additional fields
            .sim    : game simulation dictionary

    Author - Belinda Fleischmann, Dirk Ostwald
    """
    mode                = sim.mode                                              # simulation mode
    theta               = sim.theta                                             # simulation parameters

    # task, agent, and behavioral model instantiation
    t_init              = sim.t_init                                            # task initialization structure
    a_init              = sim.a_init                                            # agent initialization structure
    task                = th_task(t_init)                                       # task object
    a_init.task         = task                                                  # task embedding
    agent               = th_agent(a_init)                                      # agent initialization
    pi                  = np.array([2, 3, 4, 1])                                # TODO: decision policy, um zu zeigen, dass der agent mit vorgegebenen actions \pi rum geht 

    # task agent interaction observation
    # --------------------------------------------------------------------------
    task.start_game()                                                           # game start configuration

    for c in np.arange(theta.n_c):                                              # round iterations

        # recording arrays # TODO
        s_t           = np.full((theta.n_t + 1, theta.d_s), np.nan)                     # task state array
        o_t           = np.full((theta.n_t + 1, 1), np.nan)
        d_t           = np.full((theta.n_t + 1, 1), np.nan)                                # agent decision array
        v_t           = np.full((theta.n_t + 1, 1), np.nan)                                # agent decision array
        marg_s1_b_t   = np.full((theta.n_t + 1, 1), np.nan)                                # agent decision array
        marg_s2_b_t   = np.full((theta.n_t + 1, 1), np.nan)                                # agent decision array
        d_t           = np.full((theta.n_t + 1, 1), np.nan)                                # agent decision array
        a_t   = np.full((theta.n_t + 1, 1), np.nan)                                # action array


        # Task and agent start new round----------------------------------------
        task.c = c                                                              # round number
        task.r = 0                                                              # reward

        for t in np.arange(theta.n_t):                                          # action iterations

            # ------ TRIAL START -----------------------------------------------
            task.t = t                                                          # trial

            # Reset dynamic model components
            agent.v = np.nan                                                    # action valences
            agent.d = np.nan                                                    # decion

            # task object evaluate observation at trial start # TODO
            s_t[t, :] = task.s                                                  # task state recording

            # agent make decison
            task.identify_A_giv_s1()
            agent.delta()                                                       # agent decision
            d_t[t, :] = agent.d
            a        = agent.d                                                  # agent action
            a_t[t, :] = a

            task.f(a)                                                           # task state-state transition

    # Collect data in dataframe
    dataframe = pd.DataFrame({
        's1_t': s_t[:, 0],
        's2_t': s_t[:, 1],
        's3_1_t': s_t[:, 2],
        's3_2_t': s_t[:, 3],
        's3_3_t': s_t[:, 4],
        's3_4_t': s_t[:, 5],
        's3_5_t': s_t[:, 6],
        's3_6_t': s_t[:, 7],
        'o_t': o_t[:, 0],
        'v_t': v_t[:, 0],
        'marg_s1_b_t': marg_s1_b_t[:, 0],
        'marg_s2_b_t': marg_s2_b_t[:, 0],
        'd_t': d_t[:, 0],
        'a_t': a_t[:, 0]
    })

    # output specification
    sim.data = dataframe

    # output specification
    return sim
