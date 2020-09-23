import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
#import seaborn as sns

from visualization.src.dataproc import convert_points_to_np, convert_np_to_points, sample_data, \
    convert_examples_by_class_to_np


def plot_scatter_real_units(x, y, class_id, point_radius):
    """Normal plt.scatter uses a marker size parameter where the size is denoted
    in font pt's. However because the point_radius is a part of the inner sampling
    equation this doesn't work. The sampler expects the radius to be in the same
    units that the axes are in. This does function honors that."""
    # Adapted from https://stackoverflow.com/a/44375267
    ax = plt.gca(aspect='equal')
    assert len(x) == len(y) == len(class_id)
    colors = ['blue', 'yellow', 'green']
    fill_frac = 1
    for x, y, cl in zip(x, y, class_id):
        ax.add_artist(
            plt.Circle(
                xy=(x, y),
                facecolor=colors[cl],
                radius=point_radius*fill_frac,
                edgecolor="black",
                linestyle='-',
                linewidth=1
            )
        )


if __name__ == "__main__":
    point_radius = 0.1  # an area measure
    points, classes = make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=3,
        n_samples=300,
        n_clusters_per_class=1,
        hypercube=False
    )
    print(classes)
    #plt.scatter(points[:, 0], points[:, 1], marker='o', c=classes, s=point_size, edgecolor='k')
    plot_scatter_real_units(points[:, 0], points[:, 1], classes, point_radius)
    axes = plt.gca()
    ax_limits = [-3, 3]
    axes.set_xlim(ax_limits)
    axes.set_ylim(ax_limits)
    plt.show()
    scat_points = convert_np_to_points(points, classes)
    samples = sample_data(scat_points, 1, 1, point_radius=.2)
    for zoom_level, zoom_points_by_class in samples:
        points_sampled, classes_sampled = convert_examples_by_class_to_np(zoom_points_by_class)
        plot_scatter_real_units(
            points_sampled[:, 0], points_sampled[:, 1], classes_sampled, point_radius)
        #plt.scatter(points_sampled[:, 0], points_sampled[:, 1],
        #            marker='o', c=classes_sampled, s=point_radius, edgecolor='k')
        axes = plt.gca()
        axes.set_xlim([-3, 3])
        axes.set_ylim([-3, 3])
        plt.show()

    #is_0 = classes == 0
    #ax = sns.kdeplot(points[:, 0][is_0], points[:, 1][is_0], shade=True, legend=True)
    #plt.xlim(*ax_limits)
    #plt.ylim(*ax_limits)
    #plt.show()

