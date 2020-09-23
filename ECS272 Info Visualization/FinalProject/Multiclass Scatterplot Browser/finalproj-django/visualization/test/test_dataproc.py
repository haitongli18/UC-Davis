import random
from src.dataproc import *


p0 = lambda x, y: ScatterDataPoint(0, (x, y))
p1 = lambda x, y: ScatterDataPoint(1, (x, y))
p2 = lambda x, y: ScatterDataPoint(2, (x, y))


def test_scatter_sample():
    num_trials = 10
    for i in range(num_trials):
        sampler = ScatterDatasetSampler(
            data=(
                [p0(1, 1), p0(2, 2), p0(3, 3)],
                [p1(4, 4)],
                [p2(5, 5), p2(6, 6)],
            )
        )
        p = sampler.sample_least_filled()
        sampled_ids = {p.class_id}
        p = sampler.sample_least_filled()
        assert p.class_id not in sampled_ids
        sampled_ids.add(p.class_id)
        p = sampler.sample_least_filled()
        assert p.class_id not in sampled_ids
        sampled_ids.add(p.class_id)
        assert sampler.num_points() == (6 - 3)
        assert sampler.get_least_filled_class_ind() == 0


def test_dist_query():
    idx = DistQueriablePointTracker(3)
    idx.add_point(ScatterDataPoint(0, (1, 0)))
    assert idx.has_point_in_ball(0, (0, 0), 2)
    assert not idx.has_point_in_ball(0, (0, 0), 0.5)
    assert not idx.has_point_in_ball(1, (0, 0), 0.5)
    idx.add_point(ScatterDataPoint(0, (2, 1)))
    assert not idx.has_point_in_ball(0, (0, 0), 0.5)
    assert not idx.has_point_in_ball(0, (3, 1), 0.5)
    assert idx.has_point_in_ball(0, (2.2, 1.2), 0.5)


def test_kde():
    cl0points = [p0(1, 1), p0(1, 2), p0(1, 3), p0(5, 2)]
    cl1points = [p1(5, 9), p0(3, -2), p0(-5, 23)]
    kde = PointsKDE((cl0points, cl1points))
    assert kde.density_at(0, (2, 2)) > kde.density_at(0, (0, -5))
