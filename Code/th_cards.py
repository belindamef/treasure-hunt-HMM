from math import factorial as fact                                              # factorial function


def th_cards(theta):
    """This function computes the treasure hunt's state, observation and
    action set cardinalities

    Inputs
          theta    (obj) : task parameter structure with required fields
              .n_n (int) : number of grid world nodes/cells
              .n_h (int) : number of hiding spots

    Output
          theta       (obj) : input structure with additional fields
                .n_s3 (int) : number of unique hiding spots combination possibilities
                .n_s  (int) : state space cardinality
                .n_a  (int) : action space cardinality
                .n_i  (int) : observation space cardinality

    Authors - Belinda Fleischmann, Dirk Ostwald
    """
    # Parameters
    n_n         = theta.n_n                                                     # number of grid world nodes
    n_h         = theta.n_h                                                     # number of hiding spots

    # Cardinalities
    n_s3        = (fact(n_n)) / (fact(n_n - n_h) * fact(n_h))                   # number of unique hiding spot combination possibilities
    n_s         = n_s3 * n_h * n_n                                              # latent state space cardinality

    # Output
    theta.n_s3  = int(n_s3)                                                     # number of unique hiding spots combination possibilities
    theta.n_s   = int(n_s)                                                      # state space cardinality
    theta.n_a   = 5                                                             # action space cardinality
    theta.n_o   = 5                                                             # observation space cardinality
    return theta
