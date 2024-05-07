import os
from th_paths import th_paths
import matplotlib
import numpy as np
from matplotlib import pyplot, colors, cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle


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


def plot_agent_behavior(paths, theta, beh_data):
    """Function to plot model variables over trials

    Inputs
        theta       : task parameter structure with required fields
            .d      : dimensionality of the square grid world
            .n_n    : number of nodes
            .n_h    : number of hiding spots
            .n_s    : state space cardinality
    """

    def prepare_data() -> dict:
        """Function to prepare data for plotting in 2-dim heatmaps"""
        data = {}

        for component in model_components:
            data[component] = np.full(
                (n_nodes, n_plotable_trials + 1), np.nan)

            for trial_col in range(n_plotable_trials):

                if component in ["s1_t", "s2_t"]:
                    data[component][:, trial_col] = np.array(
                        [1 if node == beh_data[component][trial_col] - 1 else 0
                            for node in range(n_nodes)]
                    )

                # elif component == "o_t":
                #     data[component][:, trial_col] = beh_data["node_colors"][trial_col]  # TODO: hier wieter
                #     if beh_data["o_t"][trial_col][0] == 1:
                #         data[component][:, trial_col][
                #             (beh_data["s2_t"][trial_col] - 1)] = 3

                # elif component in ["marg_s1_b_t", "marg_s2_b_t"]:
                #     data[component][:, trial_col] = beh_data[component][trial_col]

                elif component in [
                    # "v_t",
                    "d_t",
                    "a_t"
                    ]:

                    # plot current position along in a grid
                    data[component][:, trial_col] = np.array(
                        [1 if node == beh_data["s1_t"][trial_col] - 1 else 0
                            for node in range(n_nodes)]
                    )
        return data

    def define_color_maps() -> dict:
        # Get viridis colormap
        viridis_cmap = cm.get_cmap("viridis")

        return {
            "s1_t": colors.ListedColormap(["black", "grey"]),
            "s2_t": colors.ListedColormap(["black", "green"]),
            "o_t": colors.ListedColormap(["black", "grey", "lightblue", "green"]),
            "marg_s1_b_t": colors.ListedColormap([viridis_cmap(0),
                                                  viridis_cmap(256)]),
            "marg_s2_b_t": "viridis",
            "v_t": colors.ListedColormap(["black", "grey"]),
            "d_t": colors.ListedColormap(["black", "grey"]),
            "a_t": colors.ListedColormap(["black", "grey"])
        }

    def define_y_labels() -> dict:
        return {
            "s1_t": r"$s^1_t$",
            "s2_t": r"$s^2_t$",
            "o_t": r"$o_t$",
            "marg_s1_b_t": r"$p(s^1_t\vert o_t)$",
            "marg_s2_b_t": r"$p(s^2_t\vert o_t)$",
            "v_t": r"$v_t$",
            "d_t": r"$d_t$",
            "a_t": r"$a_t$"
        }

    def define_variable_ranges() -> dict:
        return {
            "s1_t": [0, 1],
            "s2_t": [0, 1],
            "o_t": [0, 3],
            "marg_s1_b_t": [0, 1],
            "marg_s2_b_t": [0, 1],
            "v_t": [0, 1],
            "d_t": [0, 1],
            "a_t": [0, 1]
        }

    def define_cmap_ticks() -> dict:
        return {
            "s1_t": np.linspace(0, 1, 2),
            "s2_t": np.linspace(0, 1, 2),
            "o_t": np.linspace(0, 3, 4),
            "marg_s1_b_t": np.linspace(0, 1, 2),
            "marg_s2_b_t": np.linspace(0, 1, 2),
            "v_t": np.linspace(0, 1, 2),
            "d_t": np.linspace(0, 1, 2),
            "a_t": np.linspace(0, 1, 2)
        }

    def create_images() -> list:

        def draw_heatmap_to_this_ax() -> matplotlib.image.AxesImage:
            """Function to create heatmap Axis image from 2 dimensional
            array"""

            return this_ax.imshow(
                data[component][:, trial_col].reshape(dim, dim),
                cmap=cmaps[component],
                vmin=variable_range[component][0],
                vmax=variable_range[component][1]
            )

        def adjust_ticks_n_labels_of_this_axis():
            """Fucntion to adjust ticks, grid and labels
            """
            this_ax.set_yticklabels([])  # Remove y-axis ticks
            this_ax.set_xticklabels([])  # Remove y-axis ticks
            this_ax.set_ylabel(y_labels[component],
                               loc="center",
                               rotation="horizontal",
                               labelpad=20)
            this_ax.label_outer()
            this_ax.set_xticks(np.arange(-0.5, dim, 1), minor=True)
            this_ax.set_yticks(np.arange(-0.5, dim, 1), minor=True)
            this_ax.set_xticks([])
            this_ax.set_yticks([])
            this_ax.grid(
                which="minor",
                color='grey',
                linestyle='-',
                linewidth=0.1)

        def return_axis_coords():
            # Get the extent (x0, x1, y0, y1) in data coordinates
            extent = images[row].get_extent()  # returns the image extent as tuple (left, right, bottom, top).

            # Extract x and y coordinates
            x_coords = np.linspace(extent[0] + 1,  # left
                                   extent[1],      # right
                                   dim)
            y_coords = np.linspace(extent[3] + 1,  # top
                                   extent[2],      # bottom,
                                   dim)

            return x_coords, y_coords

        def get_node_coords(node_in_question):

            node_one_hot = np.array(
                [1 if node == node_in_question - 1 else 0
                    for node in range(n_nodes)]
            )

            node_grid_coordinate = np.where(
                node_one_hot.reshape(dim, dim) == 1)

            node_x_coord = int(node_grid_coordinate[1])  # column --> x axis
            node_y_coord = int(node_grid_coordinate[0])  # row --> y axis

            return node_x_coord, node_y_coord

        def specify_arrow_coordinates(pos_1, pos_2) -> dict:
            arrow_coords = {}

            pos_1_x, pos_1_y = get_node_coords(pos_1)
            pos_2_x, pos_2_y = get_node_coords(pos_2)

            # calculate arrow "direction" as difference between s_{t + 1} and s_t
            arrow_coords["dx"] = (axis_x_coords[pos_2_x] - 0.5) - (axis_x_coords[pos_1_x] - 0.5)
            arrow_coords["dy"] = (axis_y_coords[pos_2_y] - 0.5) - (axis_y_coords[pos_1_y] - 0.5)

            arrow_coords["start_x"] = axis_x_coords[pos_1_x] - 0.5
            arrow_coords["start_y"] = axis_y_coords[pos_1_y] - 0.5

            return arrow_coords

        def draw_arrow(arrow_coords: dict, color,
                       width_=0.002,
                       head_width=0.3,
                       head_length=0.25):

            this_ax.arrow(
                arrow_coords["start_x"],
                arrow_coords["start_y"],
                arrow_coords["dx"],
                arrow_coords["dy"],
                color=color,
                width=width_,
                length_includes_head=True,
                head_width=head_width,
                head_length=head_length)

        def draw_drill(start_x, start_y, dx, dy):

            this_ax.add_patch(
                Rectangle(
                    (start_x, start_y),
                    dx, dy,
                    # edgecolor='pink',
                    facecolor='lightgrey',
                    fill=True,
                    lw=5,
                    angle=90,
                    rotation_point="center"
                ))

        def add_colorbar():
            axs[row, -1].axis("off")                                            # remove last columns axes
            if row in [2, 3, 4]:
                divider = make_axes_locatable(axs[row, -1])
                cax = divider.append_axes("right", pad=0.0001, size="20%")
                ticks = cmap_ticks[component]

                cbar = fig.colorbar(
                    images[(row + 1) * n_plotable_trials - 1],
                    cax,
                    orientation='vertical',
                    ticks=ticks
                )

                cbar.ax.tick_params(labelsize=6)                          # Set fontsize for colorbar ticks
                # Explicitly update colorbar layout engine
                cax.get_yaxis().set_label_coords(-0.5, 0.5)
                cax.xaxis.set_label_position('top')
                cax.xaxis.set_ticks_position('top')


        # ------ Start Plotting Rountine ----------------------------------
        images = []

        # Iterate model components (rows)
        for row, component in enumerate(model_components):

            # Iterate trials (columns)
            for trial_col in range(n_plotable_trials):

                # Specify this axis
                this_ax = axs[row, trial_col]

                # Crate heatmap and append to image list
                images.append(draw_heatmap_to_this_ax())
                # Adjust ticks, labels and grid
                adjust_ticks_n_labels_of_this_axis()

                # Add arrows for action a_t
                if component in ["d_t", "a_t"]:

                    axis_x_coords, axis_y_coords = return_axis_coords()

                    # Extract action variabel for this trial
                    d_or_a = beh_data[component][trial_col]

                    # Skip routine of drawing arrow, if a_t == nan
                    if not np.isnan(d_or_a):

                        s1_t_x, s1_t_y = get_node_coords(
                            beh_data["s1_t"][trial_col]
                        )

                        if d_or_a == 0:
                            draw_drill(
                                start_x=axis_x_coords[s1_t_x] - 1,
                                start_y=axis_y_coords[s1_t_y] - 1,
                                dx=1, dy=1
                            )

                        else:  # if a_t != 0, i.e. is a step

                            arrow_coords = specify_arrow_coordinates(
                                pos_1=beh_data["s1_t"][trial_col],
                                pos_2=beh_data["s1_t"][trial_col] + d_or_a
                            )

                            draw_arrow(
                                arrow_coords=arrow_coords,
                                color=arrow_colors[component]
                            )

                # if component == "v_t":

                #     v_t = beh_data["v_t"][trial_col]
                #     if not np.any(np.isnan(v_t)):
                #         axis_x_coords, axis_y_coords = return_axis_coords()

                #         # TODO: hier weiter
                #         v_t.sort()

                #         a_giv_s1 = beh_data["a_giv_s1"][trial_col]

                #         for possible_a in a_giv_s1:

                #             arrow_coords = specify_arrow_coordinates(
                #                 pos_1=beh_data["s1_t"][trial_col],
                #                 pos_2=beh_data["s1_t"][trial_col]
                #                 + possible_a
                #             )

                #             draw_arrow(
                #                 arrow_coords=arrow_coords,
                #                 color="lightgrey"
                #            )
            add_colorbar()

        return images


    dim = theta.d                                                               # dimensionality of square grid world
    n_nodes = theta.n_n                                                         # number nodes in gridworld
    n_plotable_trials = beh_data["s1_t"].count()
    model_components = ["s1_t", "s2_t", "o_t",
                        "marg_s1_b_t", "marg_s2_b_t",
                        "v_t", "d_t",
                        "a_t"]
    n_rows = len(model_components)

    data = prepare_data()

    # ------Prepare figure-----------------------------------------------
    plt = pyplot
    rc_params = {
        'text.usetex': 'True',
        'axes.spines.top': 'False',
        'axes.spines.right': 'False',
        'yaxis.labellocation': 'bottom'
    }
    plt.rcParams.update(rc_params)

    fig, axs = plt.subplots(
        n_rows,
        n_plotable_trials + 1,
        sharex=True, sharey=True,
        # layout="constrained"
        # figsize=(9, 4)
    )

    fig.suptitle(f"Agent {beh_data['agent'][0]} behavior "
                 #r"$\tau = $" f"{beh_data['tau_gen'][0]}"
                 )

    y_labels = define_y_labels()

    cmaps = define_color_maps()
    arrow_colors = {"d_t": "lightgrey",
                    "a_t": "lightgreen"}

    variable_range = define_variable_ranges()

    cmap_ticks = define_cmap_ticks()

    images = create_images()

    fig_fn = (
        f"agent-{beh_data['agent'][0]}"
        f"_{theta.n_n}-nodes"
        f"_{theta.n_h}-hides"
    )
    fig_path = os.path.join(paths.figures, fig_fn)
    fig.savefig(f"{fig_path}.pdf",
                #dpi=200,
                format='pdf')
