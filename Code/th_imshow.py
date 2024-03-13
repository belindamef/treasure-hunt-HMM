from matplotlib import pyplot
from matplotlib.colors import ListedColormap


def plot_color_map(n_nodes, n_hides, **arrays):
    """This function visualizes 2 dimenional data on a 2D regular rastar.

    Inputs:
        n_nodes (int): number of nodes
        n_hides (int): number of hides
    """

    for key, array in arrays.items():
        fig_fn = f"Figures/{key}_{n_nodes}-nodes_{n_hides}-hides"

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
