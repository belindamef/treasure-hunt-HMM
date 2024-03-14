from math import factorial as fact                                              # factorial function


def th_cards(theta):
    """
    This function computes the treasure hunt's state, observation and
    action set cardinalities

    # TODO: cardinality of observation

    Inputs
          theta     : task parameter structure with required fields
              .n_n  : number of grid world nodes/cells
              .n_h  : number of hiding spots

    Output
          theta     : input structure with additional fields
              .n_s  : state space cardinality
              .n_a  : action space cardinality

    Authors - Belinda Fleischmann, Dirk Ostwald
    """
    # parameters
    n_n         = theta.n_n                                                     # number of grid world nodes
    n_h         = theta.n_h                                                     # number of hiding spots

    # cardinalities
    n_s3        = (fact(n_n)) / (fact(n_n - n_h) * fact(n_h))                   # number of unique hiding spot combination possibilities
    n_s         = n_s3 * n_h * n_n                                              # latent state space cardinality

    # output
    theta.n_s3  = int(n_s3)                                                     # number of unique hiding spots combination possibilities
    theta.n_s   = int(n_s)                                                      # state space cardinality
    theta.n_a   = 5                                                             # action space cardinality
    theta.n_o   = 5                                                             # observation space cardinality
    return theta
