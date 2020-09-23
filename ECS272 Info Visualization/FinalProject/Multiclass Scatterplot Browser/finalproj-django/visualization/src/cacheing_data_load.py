import os
import time
from pathlib import Path
from typing import Dict

from diskcache import Cache, FanoutCache, DjangoCache

from visualization.src.dataproc import sample_data, ColorManager
from visualization.src.datasets import get_person_activity_dataset, convert_df_to_points, \
    PersonActivityParams, DatasetParams
import pandas as pd

cur_path = Path(os.path.dirname(os.path.abspath(__file__)))
cache_path = cur_path / "../sample_cache"
cache_path.mkdir(exist_ok=True)
cache = FanoutCache(str(cache_path), timeout=1)


@cache.memoize(typed=True)
def get_person_activity_sampled_json(
    point_radius: float = .2,
    debug_subset_size: int = None,  # if set will take only this num points randomly
) -> Dict:
    """Gets the dict for the person activity dataset which could be sent to
    the webclient as a json"""
    print("-- Loading Person Activity Data")
    df = get_person_activity_dataset()
    if debug_subset_size and debug_subset_size < len(df):
        df = df.sample(debug_subset_size)
    data_dict = build_sample_dict(
        df, PersonActivityParams, point_radius=point_radius)
    return data_dict


def build_sample_dict(
    df: pd.DataFrame,
    params: DatasetParams,
    point_radius: float
) -> Dict:
    """builds a json dict for the web client for a given dataset dataframe and params"""
    scat_points = convert_df_to_points(df, params)
    print("Sampling Points:")
    st_time = time.time()
    samples = sample_data(
        data=scat_points,
        initial_zoom_level=1,
        max_zoom_level=4,
        point_radius=point_radius
    )
    end_time = time.time()
    print("Took {0: .2f} minutes to resample the points".format(end_time - st_time))

    print("Figure out Colors:")
    cm = ColorManager(scat_points)
    colors = cm.get_colors()
    print("colors: {}".format(colors))

    x = []
    y = []
    c = []
    c_name = []
    clr = []
    lvl = []
    for level_data in samples:
        cur_lvl, pts = level_data
        for cl in pts:
            for pt in cl:
                pos = pt.coord
                x.append(pos[0])
                y.append(pos[1])
                c.append(int(pt.class_id))
                c_name.append(params.class_names[int(pt.class_id)])
                clr.append(colors[int(pt.class_id)])
                lvl.append(cur_lvl)

    orig_x = list(df[params.x_coord_field])
    orig_y = list(df[params.y_coord_field])
    orig_c = [params.class_name_to_idx[class_name] for class_name in df[params.class_field]]
    orig_c_name = list(df[params.class_field])
    orig_clr = [colors[i] for i in orig_c]
    # print("x:{0}, y:{1}, c{2}, lvl:{3}".format(len(x), len(y), len(c), len(lvl)))
    data_dict = {
        'x': x,
        'y': y,
        'c': c,
        'c_name': c_name,
        'clr': clr,
        'lvl': lvl,
        'orig_x': orig_x,
        'orig_y': orig_y,
        'orig_c': orig_c,
        'orig_c_name': orig_c_name,
        'orig_clr': orig_clr
    }
    return data_dict


def precache_all():
    samples = get_person_activity_sampled_json(debug_subset_size=500)
    print(samples)


if __name__ == "__main__":
    precache_all()

