import os


class th_paths():
    """This class stores path variables to project folders on disk.

    Inputs:
        theta     : task parameter structure with fields
            .d    : dimension of the square grid world
            .n_h  : number of treasure hiding spots
    """
    def __init__(self, theta):
        d   = theta.d
        n_h = theta.n_h
        self.components = os.path.join(                                         # path to model components
            "Components", f"dim-{d}_hide-{n_h}"
        )
        self.figures    = os.path.join(                                         # path to figures
            "Figures", f"dim-{d}_hide-{n_h}"
        )
        self.make_directories()

    def make_directories(self):
        """Function to make all output directories"""
        if not os.path.exists(self.components):
            os.makedirs(self.components)
        if not os.path.exists(self.figures):
            os.makedirs(self.figures)

