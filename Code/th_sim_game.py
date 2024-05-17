import numpy as np                                                              # NumPy
import pandas as pd                                                             # Pandas
from th_model import th_model
from th_task import th_task                                                     # task model module
from th_agent import th_agent                                                   # agent model module


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
            .m_init : behavioral model initialization structure


    Outputs
        sim  : structure equivalent with additional fields
            .sim    : game simulation dictionary

    Author - Belinda Fleischmann, Dirk Ostwald
    """
    mode                = sim.mode                                              # simulation mode
    theta               = sim.theta                                             # simulation parameters

    # Task, agent, and behavioral model instantiation
    t_init              = sim.t_init                                            # task initialization structure
    a_init              = sim.a_init                                            # agent initialization structure
    m_init              = sim.m_init                                            # behavioral model initialization structure
    task                = th_task(t_init)                                       # task object
    a_init.task         = task                                                  # task embedding
    agent               = th_agent(a_init)                                      # agent initialization
    m_init.task         = task                                                  # task embedding
    m_init.agent        = agent                                                 # agent embedding
    model               = th_model(m_init)                                      # model object
    pi                  = np.array([2, 3, 4, 1])                                # TODO: decision policy, um zu zeigen, dass der agent mit vorgegebenen actions \pi rum geht 

    # Task agent interaction simulation
    # --------------------------------------------------------------------------
    task.start_game()                                                           # game start configuration

    data_one_block = pd.DataFrame()                                             # init df for one block

    for c in np.arange(theta.n_c):                                              # round iterations

        variable_list = [                                                       # list of variables to be recorded
            "s1_t",                                                             # current position
            "s2_t",                                                             # treasure location
            "s3_t",                                                             # hiding spots
            "o_t",                                                              # observation
            "v_t",                                                              # action valences
            "d_t",                                                              # agent decision
            "a_t",                                                              # action
            "r_t",                                                              # reward
            "marg_s1_b_t",                                                      # marginal belief over s1 (current position)
            "marg_s2_b_t",                                                      # marginal belief over s2 (treasure location)
            "marg_s3_b_t",                                                      # marginal belief over s3 (hiding spots)
            "node_colors"                                                       # current node colors (observable)
        ]

        round_dict = {}
        for var in variable_list:
            round_dict[var] = np.full(
                theta.n_t + 1, np.nan, dtype=object)

        data_one_round = pd.DataFrame(round_dict)

        # Task and agent start new round----------------------------------------
        task.c = c                                                              # round number
        task.r = 0                                                              # reward

        for t in np.arange(theta.n_t):                                          # action iterations

            # ------ TRIAL START -----------------------------------------------
            task.t = t                                                          # trial

            # Reset dynamic model components
            agent.v = np.nan                                                    # action valences
            agent.d = np.nan                                                    # decion

            # trial start recordings
            data_one_round.loc[t, "s1_t"] = task.s[0]                           # record task state
            data_one_round.loc[t, "s2_t"] = task.s[1]                           # record task state
            data_one_round.loc[t, "s3_t"] = task.s[2:]                          # record task state

            # agent make decison
            task.identify_A_giv_s1()                                            # evaluate set of available actions
            agent.delta()                                                       # agent decision
            # TODO: see for-deletion in agent.make_decision() for what's still missing
            data_one_round.loc[t, "d_t"] = agent.d                              # record agent decision
            a = model.return_action()                                           # agent action
            data_one_round.loc[t, "a_t"] = a                                    # record action

            # state transition
            task.f(a)                                                           # task state-state transition

        # Create a dataframe from recording array dictionary
        data_one_round.insert(0, "trial", pd.Series(                            # add trial column
            range(1, theta.n_t + 2)))
        data_one_round.insert(0, "round_", c + 1)                               # add round colunn
        data_one_block = pd.concat(                                             # append this round df to entire block df
            [data_one_block, data_one_round],
            ignore_index=True
        )

    data_one_block.insert(0, "agent", a_init.a_name)                            # add agent name column
    sim.data = data_one_block                                                   # output specification

    return sim
