import os
import csv
import scipy.sparse as sp
import pickle


class th_paths():
    """This class stores path variables to project folders on disk.

    Inputs:
        theta     : task parameter structure with fields
            .d    : dimension of the square grid world
            .n_h  : number of treasure hiding spots
    """
    def __init__(self, theta, out_directory_label):
        d            = theta.d
        n_h          = theta.n_h
        config_label = f"dim-{d}_hide-{n_h}"
        self.components = os.path.join(                                         # path to model components
            "Components", config_label
        )
        self.figures    = os.path.join(                                         # path to figures
            "Figures", f"{out_directory_label}_{config_label}"
        )
        self.data       = os.path.join(                                         # path to data
            "Data", f"{out_directory_label}_{config_label}"
        )
        self.make_directories()

    def make_directories(self):
        """Function to make all output directories"""
        if not os.path.exists(self.components):
            os.makedirs(self.components)
        if not os.path.exists(self.figures):
            os.makedirs(self.figures)

    def save_arrays(self, file_name, array, sparse=False, save_csv=False):

        if save_csv:
            # Write the vectors to a tsv file
            with open(f"{self.components}/{file_name}.csv", 'w', newline='', encoding="utf8") as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows(array)

        if sparse:
            with open(f"{self.components}/{file_name}.npz", "wb") as f:
                sp.save_npz(f, array)
        else:
            with open(f"{self.components}/{file_name}.pkl", "wb") as f:
                pickle.dump(array, f)
