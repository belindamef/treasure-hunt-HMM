import os


class th_paths():
    def __init__(self, theta, out_directory_label):
        """This function encodes the instantiation method for the path class
        to store path variables and create directories in the project folders on disk.

        Inputs
            theta               (obj) : task parameter structure with fields
                .d              (int) : dimension of the square grid world
                .n_h            (int) : number of treasure hiding spots
            out_directory_label (str) : label for output directories /Data and /Figures

        Outputs
            NONE

        Creates on disk, if not existing
            /Components directory     : to store arrays of S, A, O, Phi and Omega
            /Figures directory        : to store figures
            /Data directory           : to store simulated data
        """
        d               = theta.d                                               # grid dimension
        n_h             = theta.n_h                                             # number of hiding spots
        config_label    = f"dim-{d}_hide-{n_h}"
        self.components = os.path.join(                                         # path to model components
            "Components", config_label)
        self.figures    = os.path.join(                                         # path to figures
            "Figures", f"{out_directory_label}_{config_label}")
        self.data       = os.path.join(                                         # path to data
            "Data", f"{out_directory_label}_{config_label}")
        self.make_directories()

    def make_directories(self):
        """Function to make all output directories

        Inputs
            self            (obj) : paths object
                .components (str) : paths to components directory
                .figures    (str) : paths to figures directory
                .data       (str) : paths to data directory

        Creates on disk, if not existing
            /Components directory     : to store arrays of S, A, O, Phi and Omega
            /Figures directory        : to store figures
            /Data directory           : to store simulated data
        """
        if not os.path.exists(self.components):
            os.makedirs(self.components)                                        # Make /Components
        if not os.path.exists(self.figures):
            os.makedirs(self.figures)                                           # Make /Figures
        if not os.path.exists(self.data):
            os.makedirs(self.data)                                              # Make /Data
