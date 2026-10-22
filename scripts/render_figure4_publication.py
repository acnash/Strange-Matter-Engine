"""Publication-only restyling of Figure 4 using the original projections."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import NullFormatter
import render_four_dynamical_behaviours as source


def main():
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 23,
                         'text.color': 'black', 'axes.labelcolor': 'black',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    coordinates = {
        'point': source.pca_project(source.SOURCES['point'], slice(0, 500), terminal_center=True),
        'strange': source.pca_project(source.SOURCES['strange'], circular=True),
        'complex': source.pca_project(source.SOURCES['complex'], circular=True),
        'oscillator': source.oscillator_delay_project(source.SOURCES['oscillator']),
    }
    fig = plt.figure(figsize=(18, 16), facecolor='white')
    specifications = [
        ('point', 'A) Point attractor', (22, 42), slice(None), False),
        ('strange', 'B) Hyperchaotic candidate', (25, 165), slice(None), False),
        ('complex', 'C) Persistent complex variation', (24, 138), slice(None), False),
        ('oscillator', 'D) Period-two oscillator candidate', (24, 152), slice(1000, None), True),
    ]
    for i, (key, title, view, window, phases) in enumerate(specifications):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        source.add_static_panel(ax, coordinates[key], title, '', view, window,
                                .35 if phases else 1.0, phases)
        for txt in list(ax.texts):
            txt.remove()
        ax.text2D(.5, 1.0, title, transform=ax.transAxes, ha='center', va='top',
                  fontsize=25, fontweight='bold', color='black')
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_formatter(NullFormatter())
            axis._axinfo['tick']['inward_factor'] = 0
            axis._axinfo['tick']['outward_factor'] = 0
        # The original integer colormap argument selected the second colour;
        # use the endpoint of the continuous generation scale for generation 5000.
        ax.collections[2].set_facecolor(plt.colormaps['viridis'](1.0))
        ax.legend(loc='upper left', bbox_to_anchor=(.0, .91), frameon=False,
                  fontsize=22, labelspacing=.45, handletextpad=.35, labelcolor='black')
    fig.subplots_adjust(left=.01, right=.86, bottom=.015, top=.99, hspace=.05, wspace=.015)
    cax = fig.add_axes([.9, .24, .018, .52])
    norm = colors.PowerNorm(gamma=.35, vmin=0, vmax=5000)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'), cax=cax)
    cb.set_ticks([0, 25, 50, 100, 250, 500, 1000, 2500, 5000])
    cb.ax.tick_params(labelsize=23, colors='black')
    cb.set_label('Cellular automata generation', fontsize=23, labelpad=15, color='black')
    output = source.FIGURES / '17_four_graph_ca_dynamical_behaviours'
    fig.savefig(output.with_suffix('.png'), dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(output.with_name(output.name + '_preview.png'), dpi=100,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(output.with_suffix('.png'))


if __name__ == '__main__':
    main()
