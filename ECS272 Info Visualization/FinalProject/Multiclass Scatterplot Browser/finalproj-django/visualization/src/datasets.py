"""A python wrapper for loading different used datasets"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from visualization.src.dataproc import EXAMPLES_BY_CLASS, convert_np_to_points
import numpy as np
import os
from visualization.src.util import cached_property


@dataclass()
class DatasetParams:
    """Used to the """
    class_names: List[str]
    x_coord_field: str
    """The name of the field in the dataframe representing the x coord"""
    y_coord_field: str
    """The name of the field in the dataframe representing the y coord"""
    class_field: str
    """The name of the field in the dataframe representing the class name"""

    def __post_init__(self):
        self.class_name_to_idx = {n: i for i, n in enumerate(self.class_names)}


PersonActivityParams = DatasetParams(
    class_names=['walking', 'sitting', 'standing up from lying'],
    x_coord_field="X",
    y_coord_field="Y",
    class_field="Activity"
)


def get_person_activity_dataset() -> pd.DataFrame:
    cur_file = Path(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(cur_file / '../../ConfLongDemo_JSI.csv')
    filter_locs = df[PersonActivityParams.class_field] == PersonActivityParams.class_names[0]
    for cl_name in PersonActivityParams.class_names[1:]:
        filter_locs |= df[PersonActivityParams.class_field] == cl_name
    filtered = df.loc[filter_locs]
    return filtered


def convert_df_to_points(
    df: pd.DataFrame,
    params: DatasetParams
) -> EXAMPLES_BY_CLASS:
    points = np.vstack([df[params.x_coord_field], df[params.y_coord_field]]).T
    classes = np.array([params.class_name_to_idx[a] for a in df[params.class_field]])
    return convert_np_to_points(points, classes)


# sample generated test data
# ----------------------------------------------------------------
# points, classes = make_classification(
#     n_features=2,
#     n_redundant=0,
#     n_informative=2,
#     n_classes=2,
#     n_samples=300,
#     n_clusters_per_class=1,
#     hypercube=False
# )
# scat_points = convert_np_to_points(points, classes)
# samples = sample_data(scat_points, 1, 3, point_radius=.2)
# ----------------------------------------------------------------


