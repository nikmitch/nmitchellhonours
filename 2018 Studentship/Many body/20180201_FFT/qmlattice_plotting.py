#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
import os
import time

# Complementary, and nearly standard libraries
import matplotlib.pyplot as plt

# == FUNCTION DECLARATIONS =====================================================
def nik_plot(fk_zm, parameters):
    '''Moved it here as your snippet was temporary in the evolve function.'''
    tmin = parameters["Time starts"]
    tmax = parameters["Time ends"]
    tn = parameters["Number of time steps"]

    # Time vector for plotting
    times = np.linspace(tmin, tmax, tn)

    plt.plot(times, fk_zm)
    plt.title("Zero-momentum as time evolves, (U = 0)")
    plt.show()
    return


def plot_lattice(parameters):
    '''Plot the lattice and all connections'''
    nx = parameters["Number of sites per chain"]
    ny = parameters["Number of chains"]

    mapping = site2index(nx, ny)
    inverse = {v: k for (k, v) in mapping.items()}

    # Creating the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Links between the chains
    for (site1, site2) in parameters["Links between chains"]:
        # The strange order lies in the numbering scheme we settled on
        (y1, x1) = inverse[site1]
        (y2, x2) = inverse[site2]

        # Shift is necessary as we start numbering from the top chain
        y1 = ny - (y1 + 1)
        y2 = ny - (y2 + 1)

        plt.plot([x1, x2], [y1, y2], 'r--', linewidth=2)

    # Links within the chains
    for (site1, site2) in parameters["Links within chains"]:
        # The strange order lies in the numbering scheme we settled on
        (y1, x1) = inverse[site1]
        (y2, x2) = inverse[site2]

        # Shift is necessary as we start numbering from the top chain
        y1 = ny - (y1 + 1)
        y2 = ny - (y2 + 1)

        plt.plot([x1, x2], [y1, y2], 'b-', linewidth=2)

    # Blob for all sites
    for site in mapping:
        (y1, x1) = site
        label = str(mapping[site])

        # Shift is necessary as we start numbering from the top chain
        y1 = ny - (y1 + 1)

        # All sites
        plt.plot(x1, y1, 'bo', markersize=10)
        ax.annotate(label, xy=(x1, y1), xytext=(x1+0.05, y1+0.05), size=15)

    # Global settings for the graph
    plt.grid(True)                                 # Turn grid on
    plt.setp(ax.get_xticklabels(), visible=False)  # Turn xticks and labels off
    plt.setp(ax.get_yticklabels(), visible=False)  # Turn yticks and labels off
    plt.xlim([-0.5, nx-0.5])                       # Giving some margin in x
    plt.ylim([-0.5, ny-0.5])                       # Giving some margin in y
    plt.close(fig)                                 # Close figure, so no wait

    # Full path of the graphics file
    directory = os.getcwd()
    timestamp = parameters["Timestamp"]
    figfname = "".join([timestamp, "_geometry.png"])
    figfname = os.path.normpath("".join([directory, "/", figfname]))

    # Save figure
    fig.savefig(figfname, paper="a4", orientation="landscape", dpi=300,
                bbox_inches="tight")
    return

# ==============================================================================
if __name__ == "__main__":
    pass