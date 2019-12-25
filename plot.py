import contextlib
import random
from matplotlib import pyplot
import seaborn

pyplot.style.use('seaborn-whitegrid')
pyplot.rcParams.update({'font.size': 15})


@contextlib.contextmanager
def gfx_setup(filename, xlabel='Ability with thing A', ylabel='Ability with thing B'):
    fig, axes = pyplot.subplots(2, 2, sharex='col', sharey='row',
                                gridspec_kw={'width_ratios': [3, 1],
                                             'height_ratios': [1, 3],
                                             'wspace': 0,
                                             'hspace': 0})
    for i, j in [(0, 0), (0, 1), (1, 1)]:
        axes[i][j].axis('off')
    ax = axes[1][0]
    fig.set_size_inches(7, 7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal', 'box')
    ax.set_xticks(range(-3, 4))
    ax.set_yticks(range(-3, 4))
    ax.grid(which='major', alpha=0.3)
    yield fig, axes
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', frameon=True, facecolor='white', bbox_to_anchor=(1, 1))
    # fig.tight_layout()
    fig.savefig(filename)


xs = [random.gauss(0, 1) for i in range(5000)]
ys = [random.gauss(0, 1) for i in range(5000)]

def scatter(axes, fn, color, label):
    axes[1][0].scatter([], [], color=color, label=label)  # just for label
    axes[1][0].scatter([x for x, y in zip(xs, ys) if fn(x, y)],
                       [y for x, y in zip(xs, ys) if fn(x, y)],
                       alpha=0.1, color=color, edgecolors='none')
    axes[0][0].hist([x for x, y in zip(xs, ys) if fn(x, y)],
                    bins=10, alpha=0.3, color=color, density=True)
    axes[1][1].hist([y for x, y in zip(xs, ys) if fn(x, y)],
                    bins=10, alpha=0.3, color=color, density=True, orientation='horizontal')

with gfx_setup('plot.png') as (fig, axes):
    scatter(axes, lambda x, y: True,
            color='b', label='Candidates we consider')

with gfx_setup('plot2.png') as (fig, axes):
    scatter(axes, lambda x, y: x+y > 0.5,
            color='g', label='Candidates we bring in')
    scatter(axes, lambda x, y: x+y <= 0.5,
            color='r', label='Candidates we do not bring in')
    #pyplot.plot([-10, 10], [10.5, -9.5], color='black')

with gfx_setup('plot3.png') as (fig, axes):
    scatter(axes, lambda x, y: x+y > 1.5,
            color=(1, 0.5, 0), label='Candidates that do not want to talk to us')
    scatter(axes, lambda x, y: 0.5 < x+y <= 1.5,
            color='g', label='Candidates we bring in')
    scatter(axes, lambda x, y: x+y <= 0.5,
            color='r', label='Candidates we do not bring in')
    #pyplot.plot([-10, 10], [10.5, -9.5], color='black')
    #pyplot.plot([-10, 10], [11.5, -8.5], color='black')

with gfx_setup('plot_confidence.png', xlabel='Competence', ylabel='Confidence') as (fig, axes):
    scatter(axes, lambda x, y: x+y > 1.5,
            color=(1, 0.5, 0), label='Candidates that do not want to talk to us')
    scatter(axes, lambda x, y: 1 < x and x+y <= 1.5,
            color='g', label='Candidates we bring in')
    scatter(axes, lambda x, y: x <= 1 and x+y <= 1.5,
            color='r', label='Candidates we do not bring in')
    #pyplot.plot([-10, 10], [11.5, -8.5], color='black')
    #pyplot.plot([1, 1], [0.5, -10], color='black')

with gfx_setup('plot_fancy_school.png', xlabel='Did well on coding test', ylabel='Went to fancy school') as (fig, axes):
    scatter(axes, lambda x, y: x+y > 1.5,
            color=(1, 0.5, 0), label='Candidates that do not want to talk to us')
    scatter(axes, lambda x, y: 2 < 2*x+y and x+y <= 1.5,
            color='g', label='Candidates we bring in')
    scatter(axes, lambda x, y: 2*x+y <= 2 and x+y <= 1.5,
            color='r', label='Candidates we do not bring in')
    #pyplot.plot([-10, 10], [11.5, -8.5], color='black')
    #pyplot.plot([1, 1], [0.5, -10], color='black')
