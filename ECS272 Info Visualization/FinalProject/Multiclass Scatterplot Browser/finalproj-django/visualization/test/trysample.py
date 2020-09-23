from src.dataproc import *

if __name__ == "__main__":
    p0 = lambda x, y: ScatterDataPoint(0, (x, y))
    p1 = lambda x, y: ScatterDataPoint(1, (x, y))
    p2 = lambda x, y: ScatterDataPoint(2, (x, y))
    data=(
        [p0(1, 1), p0(2, 2), p0(3, 3)],
        [p1(4, 4), p1(5, 5)],
        [p2(6, 6), p2(7, 7)],
    )
    sample_data(data, 1, 3, 15)