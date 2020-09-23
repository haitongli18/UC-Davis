from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
import pandas as pd
import numpy as np
import time
#import seaborn as sns
from visualization.src.cacheing_data_load import get_person_activity_sampled_json
from visualization.src.dataproc import convert_points_to_np, convert_np_to_points, sample_data, \
    convert_examples_by_class_to_np, ColorManager


def __test__(request):
    return render(request, 'visualization.html')


def get_scatter_points(request):
    """
    # Return resampled scatter points in json to the frontend.
    Format: `{x: List[int], y: List[int], c: List[int], c_name: List[str], color: List[int], lvl: List[int]}`

    url: `get_scatter_points/`

    ~Currently a test without zoom-in feature~
    """
    # For debugging we won't use all the points
    debug_subset_size: int = None  # Set to None to include all points

    data = get_person_activity_sampled_json(
        point_radius=0.05,  # This might need to be adjusted.
        debug_subset_size=debug_subset_size
    )
    return JsonResponse(data)


def get_orig_scatter_points(request):
    """
    # Return original scatter points in json to the frontend.
    Format: `{x: List[int], y: List[int], c: List[int], color: List[int], lvl: List[int]}`

    url: `get_orig_scatter_points/`

    ~Currently a test without zoom-in feature~
    """

    df = pd.read_csv('ConfLongDemo_JSI.csv')
    filtered = df.loc[(df['Activity'] == 'walking') | (df['Activity'] == 'sitting') | (df['Activity'] == 'standing up from lying')]
    print("Number of instances: {0}".format(len(filtered)))

    class_mapping = {'walking': 0, 'sitting': 1, 'standing up from lying': 2}

    points = np.vstack([filtered.X, filtered.Y]).T
    classes = np.array([class_mapping[a] for a in filtered.Activity])
    # to get the colors we do need KDE, so it's still necessary to convert them to points
    scat_points = convert_np_to_points(points, classes)

    cm = ColorManager(scat_points)
    colors = cm.get_colors()

    x = list(filtered.X)
    y = list(filtered.Y)
    c = [class_mapping[cl] for cl in filtered.Activity]
    clr = [colors[i] for i in c]
    lvl = [1 for i in x]

    data = {'x': x, 'y': y, 'c': c, 'clr': clr, 'lvl': lvl}
    return JsonResponse(data)