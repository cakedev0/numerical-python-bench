# bench_sklearn.py
import argparse, contextlib, time, os
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from sklearn.datasets import make_classification, make_blobs
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting import gradient_boosting as hgbt_module
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument("--bench", choices=["hgbt", "kmeans"], required=True)
parser.add_argument("--threads", type=int, default=None)
args = parser.parse_args()

print("ENV:", {k: os.environ.get(k) for k in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "OMP_PROC_BIND", "OMP_PLACES"
]})
print(threadpool_info())


class HGBTTimer:
    def __init__(self):
        self.binning_train = 0.0
        self.binning_validation = 0.0
        self.grower_init = 0.0
        self.grow = 0.0
        self.make_predictor = 0.0
        self.compute_hist = 0.0
        self.find_split = 0.0
        self.apply_split = 0.0
        self.n_growers = 0
        self.n_nodes = 0

    @property
    def binning(self):
        return self.binning_train + self.binning_validation

    @property
    def tree_total(self):
        return self.grower_init + self.grow + self.make_predictor


@contextlib.contextmanager
def instrument_hgbt(est):
    timer = HGBTTimer()
    original_bin_data = est._bin_data
    original_tree_grower = hgbt_module.TreeGrower

    def timed_bin_data(X, sample_weight, is_training_data):
        t0 = time.perf_counter()
        result = original_bin_data(X, sample_weight, is_training_data)
        dt = time.perf_counter() - t0
        if is_training_data:
            timer.binning_train += dt
        else:
            timer.binning_validation += dt
        return result

    class TimedTreeGrower(original_tree_grower):
        def __init__(self, *args, **kwargs):
            t0 = time.perf_counter()
            super().__init__(*args, **kwargs)
            timer.grower_init += time.perf_counter() - t0
            timer.n_growers += 1

        def grow(self):
            t0 = time.perf_counter()
            try:
                return super().grow()
            finally:
                timer.grow += time.perf_counter() - t0
                timer.compute_hist += self.total_compute_hist_time
                timer.find_split += self.total_find_split_time
                timer.apply_split += self.total_apply_split_time
                timer.n_nodes += self.n_nodes

        def make_predictor(self, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return super().make_predictor(*args, **kwargs)
            finally:
                timer.make_predictor += time.perf_counter() - t0

    est._bin_data = timed_bin_data
    hgbt_module.TreeGrower = TimedTreeGrower
    try:
        yield timer
    finally:
        est._bin_data = original_bin_data
        hgbt_module.TreeGrower = original_tree_grower


def fmt_seconds(value):
    return f"{value:9.3f}s"


def fmt_percent(value, total):
    if total <= 0:
        return "   n/a "
    return f"{100 * value / total:6.1f}%"


def print_timing_line(label, value, total, indent=2):
    print(f"{' ' * indent}{label:<18} {fmt_seconds(value)}  {fmt_percent(value, total)}")


def print_hgbt_timings(run_idx, max_bins, total, timer):
    non_binning = total - timer.binning
    non_tree = total - timer.binning - timer.tree_total

    print(f"\nhgbt_fit run={run_idx} max_bins={max_bins} trees={timer.n_growers} nodes={timer.n_nodes}")
    print_timing_line("total", total, total)
    print_timing_line("binning", timer.binning, total)
    print_timing_line("after_binning", non_binning, total)
    print_timing_line("tree_total", timer.tree_total, total)
    print_timing_line("other", non_tree, total)
    print("  tree details")
    print_timing_line("grower_init", timer.grower_init, total, indent=4)
    print_timing_line("grow", timer.grow, total, indent=4)
    print_timing_line("make_predictor", timer.make_predictor, total, indent=4)
    print("  grow details")
    print_timing_line("hist", timer.compute_hist, total, indent=4)
    print_timing_line("find_split", timer.find_split, total, indent=4)
    print_timing_line("apply_split", timer.apply_split, total, indent=4)


with threadpool_limits(args.threads):
    if args.bench == "hgbt":
        X, y = make_classification(
            n_samples=100_000,
            n_features=100,
            n_informative=5,
            random_state=0,
        )
        for i in range(3):
            max_bins = 255 - i
            est = HistGradientBoostingClassifier(
                max_iter=100,
                max_leaf_nodes=31,
                random_state=0,
                max_features=.5,
                max_bins=max_bins
            )
            t0 = time.perf_counter()
            with instrument_hgbt(est) as timer:
                est.fit(X, y)
            print_hgbt_timings(i + 1, max_bins, time.perf_counter() - t0, timer)
    
    elif args.bench == "kmeans":
        X, _ = make_blobs(
            n_samples=1_000_000,
            n_features=100,
            centers=50,
            random_state=0,
        )
        for _ in range(3):
            est = KMeans(n_clusters=50, n_init=1, algorithm="lloyd", random_state=0)
            t0 = time.perf_counter()
            est.fit(X)
            print("kmeans_fit", time.perf_counter() - t0)
