import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import numpy as np


def make_main_icon():
    n_turns = 1.5
    fig = plt.figure(frameon=False)
    fig.set_size_inches(1, 1)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    x = [0, 1, 2, 3, 2]
    y = [1, 3, 2, 3, 0]

    spline_effect = [
        pe.Stroke(linewidth=12, foreground="white"),
        pe.Stroke(linewidth=6, foreground="cornflowerblue"),
    ]

    patch = mp.patches.Rectangle([0, 0], width=1, height=1, facecolor='midnightblue',
                                edgecolor='white', linewidth=10,
                                transform=ax.transAxes)
    ax.add_patch(patch)

    ax.plot(x, y, 'o', color='cornflowerblue', mec='white', mew=3, ms=10)
    #ys = np.mean(y)
    #ax.axis(xmin=-b+xs, xmax=b+xs, ymin=-b+ys, ymax=b+ys)
    ax.axis(xmin=-1.5, xmax=4.5, ymin=-1.5, ymax=4.5)
    fig.savefig('main-icon.png', transparent=False, bbox_inches='tight')


make_main_icon()
