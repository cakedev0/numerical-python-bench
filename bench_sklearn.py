# bench_sklearn.py
import argparse, contextlib, gc, time, os
from threadpoolctl import threadpool_info, threadpool_limits
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting import gradient_boosting as hgbt_module

parser = argparse.ArgumentParser()
parser.add_argument("--problem", choices=["clf", "reg"], default="clf")
parser.add_argument("--threads", type=int, default=None)
parser.add_argument("--disable-gc", action="store_true")
args = parser.parse_args()

if False:
    print("ENV:", {k: os.environ.get(k) for k in [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "OMP_PROC_BIND", "OMP_PLACES"
    ]})
    print(threadpool_info())


class HGBTTimer:
    def __init__(self):
        self.binning_train = 0.0
        self.binning_validation = 0.0
        self.bin_fit = 0.0
        self.bin_transform = 0.0
        self.grower_init = 0.0
        self.grow = 0.0
        self.make_predictor = 0.0
        self.compute_hist = 0.0
        self.find_split = 0.0
        self.apply_split = 0.0
        self.gc_total = 0.0
        self.gc_fit = 0.0
        self.gc_transform = 0.0
        self.gc_tree = 0.0
        self.gc_other = 0.0
        self.gc_collections = 0
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
    current_phase = {"name": "other"}
    gc_start = {"time": None, "phase": None}

    def gc_callback(phase, info):
        if phase == "start":
            gc_start["time"] = time.perf_counter()
            gc_start["phase"] = current_phase["name"]
        elif phase == "stop" and gc_start["time"] is not None:
            dt = time.perf_counter() - gc_start["time"]
            timer.gc_total += dt
            timer.gc_collections += 1
            if gc_start["phase"] == "fit":
                timer.gc_fit += dt
            elif gc_start["phase"] == "transform":
                timer.gc_transform += dt
            elif gc_start["phase"] == "tree":
                timer.gc_tree += dt
            else:
                timer.gc_other += dt
            gc_start["time"] = None
            gc_start["phase"] = None

    @contextlib.contextmanager
    def phase(name):
        previous = current_phase["name"]
        current_phase["name"] = name
        try:
            yield
        finally:
            current_phase["name"] = previous

    def timed_bin_data(X, sample_weight, is_training_data):
        bin_mapper = est._bin_mapper
        original_fit = bin_mapper.fit
        original_transform = bin_mapper.transform

        def timed_fit(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                with phase("fit"):
                    return original_fit(*args, **kwargs)
            finally:
                timer.bin_fit += time.perf_counter() - t0

        def timed_transform(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                with phase("transform"):
                    return original_transform(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                timer.bin_transform += dt

        bin_mapper.fit = timed_fit
        bin_mapper.transform = timed_transform
        t0 = time.perf_counter()
        try:
            result = original_bin_data(X, sample_weight, is_training_data)
            dt = time.perf_counter() - t0
            if is_training_data:
                timer.binning_train += dt
            else:
                timer.binning_validation += dt
            return result
        finally:
            bin_mapper.fit = original_fit
            bin_mapper.transform = original_transform

    class TimedTreeGrower(original_tree_grower):
        def __init__(self, *args, **kwargs):
            t0 = time.perf_counter()
            with phase("tree"):
                super().__init__(*args, **kwargs)
            timer.grower_init += time.perf_counter() - t0
            timer.n_growers += 1

        def grow(self):
            t0 = time.perf_counter()
            try:
                with phase("tree"):
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
    gc.callbacks.append(gc_callback)
    try:
        yield timer
    finally:
        with contextlib.suppress(ValueError):
            gc.callbacks.remove(gc_callback)
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


def print_hgbt_timings(problem, run_idx, max_bins, total, timer):
    non_binning = total - timer.binning
    non_tree = total - timer.binning - timer.tree_total

    print(f"\nhgbt_{problem} run={run_idx} max_bins={max_bins} trees={timer.n_growers} nodes={timer.n_nodes}")
    print_timing_line("total", total, total)
    print_timing_line("other", non_tree + timer.make_predictor, total)
    print_timing_line("binning", timer.binning, total)
    print_timing_line("fit", timer.bin_fit, total, indent=4)
    print_timing_line("transform", timer.bin_transform, total, indent=4)
    if timer.gc_total:
        print_timing_line("gc_total", timer.gc_total, total)
        print_timing_line("gc_fit", timer.gc_fit, total, indent=4)
        print_timing_line("gc_transform", timer.gc_transform, total, indent=4)
        print_timing_line("gc_tree", timer.gc_tree, total, indent=4)
        print_timing_line("gc_other", timer.gc_other, total, indent=4)
        print(f"    gc_collections     {timer.gc_collections:9d}")
    print_timing_line("tree", timer.tree_total, total)
    print_timing_line("hist", timer.compute_hist, total, indent=4)
    print_timing_line("find_split", timer.find_split, total, indent=4)
    print_timing_line("apply_split", timer.apply_split, total, indent=4)


@contextlib.contextmanager
def maybe_disable_gc(disable):
    if not disable:
        yield
        return

    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


with threadpool_limits(args.threads):
    with maybe_disable_gc(args.disable_gc):
        if args.problem == "clf":
            X, y = make_classification(
                n_samples=100_000,
                n_features=100,
                n_informative=5,
                random_state=0,
            )
            estimator_cls = HistGradientBoostingClassifier
        else:
            X, y = make_regression(
                n_samples=100_000,
                n_features=100,
                n_informative=5,
                random_state=0,
            )
            estimator_cls = HistGradientBoostingRegressor

        for i in range(3):
            max_bins = 255
            est = estimator_cls(
                max_iter=100,
                max_leaf_nodes=31,
                random_state=0,
                max_features=.5,
                max_bins=max_bins,
                early_stopping=False,
            )
            t0 = time.perf_counter()
            with instrument_hgbt(est) as timer:
                est.fit(X, y)
            print_hgbt_timings(args.problem, i + 1, max_bins, time.perf_counter() - t0, timer)
