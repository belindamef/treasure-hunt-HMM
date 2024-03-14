import os
from th_paths import th_paths
from matplotlib import pyplot
from matplotlib.colors import ListedColormap


def plot_color_map(paths: th_paths, **arrays):
    """This function visualizes 2 dimenional data on a 2D regular rastar.

    Inputs:
        paths           : class with paths variables
        **arrays (dict) : Keyword arguments, 2-dimensional arrays to be plotted
                          Keys represent name of matrix, and values represent corresponding arrays.

    """

    for key, array in arrays.items():
        fig_fn = os.path.join(paths.figures, key)

        # Preapre figure
        plt = pyplot

        fig, ax = plt.subplots(figsize=(11, 5))

        # Create a custom discrete colormap
        cmap = ListedColormap(['darkgrey', 'darkcyan'])
        image = ax.matshow(array, cmap=cmap)

        # Add colorbar
        plt.colorbar(image, ticks=[0, 1], shrink=0.4)

        # Save or display the plot
        fig.savefig(f"{fig_fn}.pdf", format='pdf')
