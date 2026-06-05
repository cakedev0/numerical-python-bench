#!/usr/bin/env python
"""Run a predefined HGBT scaling sweep and write JSON/text results."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import statistics
import time

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting import gradient_boosting as hgbt_module
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class Workload:
    name: str
    problem: str
    n_samples: int
    n_features: int
    n_informative: int
    max_iter: int
    max_leaf_nodes: int
    max_features: float
    max_bins: int = 255


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
        self.n_growers = 0
        self.n_nodes = 0

    @property
    def binning(self):
        return self.binning_train + self.binning_validation

    @property
    def tree(self):
        return self.grower_init + self.grow + self.make_predictor


@contextlib.contextmanager
def instrument_hgbt(est):
    timer = HGBTTimer()
    original_bin_data = est._bin_data
    original_tree_grower = hgbt_module.TreeGrower

    def timed_bin_data(X, sample_weight, is_training_data):
        bin_mapper = est._bin_mapper
        original_fit = bin_mapper.fit
        original_transform = bin_mapper.transform

        def timed_fit(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original_fit(*args, **kwargs)
            finally:
                timer.bin_fit += time.perf_counter() - t0

        def timed_transform(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original_transform(*args, **kwargs)
            finally:
                timer.bin_transform += time.perf_counter() - t0

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


@contextlib.contextmanager
def disabled_gc():
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def make_data(workload):
    if workload.problem == "clf":
        return make_classification(
            n_samples=workload.n_samples,
            n_features=workload.n_features,
            n_informative=workload.n_informative,
            random_state=0,
        )
    return make_regression(
        n_samples=workload.n_samples,
        n_features=workload.n_features,
        n_informative=workload.n_informative,
        random_state=0,
    )


def make_estimator(workload):
    estimator_cls = (
        HistGradientBoostingClassifier
        if workload.problem == "clf"
        else HistGradientBoostingRegressor
    )
    return estimator_cls(
        max_iter=workload.max_iter,
        max_leaf_nodes=workload.max_leaf_nodes,
        random_state=0,
        max_features=workload.max_features,
        max_bins=workload.max_bins,
        early_stopping=False,
    )


def run_one(workload, threads, repeat):
    X, y = make_data(workload)
    with threadpool_limits(threads):
        est = make_estimator(workload)
        t0 = time.perf_counter()
        with instrument_hgbt(est) as timer:
            est.fit(X, y)
        total = time.perf_counter() - t0

    return {
        "workload": asdict(workload),
        "threads": threads,
        "repeat": repeat,
        "total": total,
        "other": total - timer.binning - timer.tree + timer.make_predictor,
        "binning": timer.binning,
        "bin_fit": timer.bin_fit,
        "bin_transform": timer.bin_transform,
        "tree": timer.tree,
        "grower_init": timer.grower_init,
        "grow": timer.grow,
        "make_predictor": timer.make_predictor,
        "hist": timer.compute_hist,
        "find_split": timer.find_split,
        "apply_split": timer.apply_split,
        "trees": timer.n_growers,
        "nodes": timer.n_nodes,
    }


def median_row(rows, key):
    return statistics.median(row[key] for row in rows)


def summarize(results):
    grouped = {}
    for row in results:
        key = (row["workload"]["name"], row["threads"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (name, threads), rows in sorted(grouped.items()):
        summary.append(
            {
                "workload": name,
                "problem": rows[0]["workload"]["problem"],
                "threads": threads,
                "repeats": len(rows),
                "total": median_row(rows, "total"),
                "binning": median_row(rows, "binning"),
                "bin_fit": median_row(rows, "bin_fit"),
                "bin_transform": median_row(rows, "bin_transform"),
                "tree": median_row(rows, "tree"),
                "hist": median_row(rows, "hist"),
                "find_split": median_row(rows, "find_split"),
                "apply_split": median_row(rows, "apply_split"),
                "nodes": median_row(rows, "nodes"),
            }
        )
    return summary


def write_text_summary(path, summary):
    by_workload = {}
    for row in summary:
        by_workload.setdefault(row["workload"], []).append(row)

    with path.open("w") as f:
        for workload, rows in by_workload.items():
            rows = sorted(rows, key=lambda row: row["threads"])
            base = rows[0]["total"]
            f.write(f"\n## {workload} ({rows[0]['problem']})\n")
            f.write(
                "threads total_s speedup binning_s bin_fit_s bin_transform_s "
                "tree_s hist_s find_s apply_s nodes\n"
            )
            for row in rows:
                f.write(
                    f"{row['threads']:7d} "
                    f"{row['total']:7.3f} "
                    f"{base / row['total']:7.2f} "
                    f"{row['binning']:9.3f} "
                    f"{row['bin_fit']:9.3f} "
                    f"{row['bin_transform']:15.3f} "
                    f"{row['tree']:7.3f} "
                    f"{row['hist']:6.3f} "
                    f"{row['find_split']:6.3f} "
                    f"{row['apply_split']:7.3f} "
                    f"{row['nodes']:5.0f}\n"
                )


WORKLOADS = [
    Workload("baseline_clf", "clf", 100_000, 100, 5, 100, 31, 0.5),
    Workload("low_features_per_split", "clf", 100_000, 100, 5, 100, 31, 0.1),
    Workload("all_features_per_split", "clf", 100_000, 100, 5, 100, 31, 1.0),
    Workload("stumps_leaf2", "clf", 100_000, 100, 5, 100, 2, 0.5),
    Workload("large_leaf255", "clf", 100_000, 100, 5, 50, 255, 0.5),
    Workload("huge_leaf1000", "clf", 60_000, 100, 5, 20, 1000, 0.5),
    Workload("few_features", "clf", 100_000, 10, 5, 100, 31, 1.0),
    Workload("many_features", "clf", 30_000, 1000, 20, 50, 31, 0.1),
    Workload("baseline_reg", "reg", 100_000, 100, 5, 100, 31, 0.5),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/hgbt_sweep")
    parser.add_argument("--threads", default="1,4,16,32,64")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    threads = [int(value) for value in args.threads.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.jsonl"
    summary_json_path = output_dir / "summary.json"
    summary_txt_path = output_dir / "summary.txt"

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata = {
        "started": started,
        "threads": threads,
        "repeats": args.repeats,
        "gc_disabled": True,
        "env": {
            key: os.environ.get(key)
            for key in ["OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES"]
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    results = []
    with disabled_gc(), raw_path.open("w") as raw_file:
        for workload in WORKLOADS:
            for thread_count in threads:
                for repeat in range(1, args.repeats + 1):
                    print(
                        f"{workload.name} threads={thread_count} repeat={repeat}",
                        flush=True,
                    )
                    row = run_one(workload, thread_count, repeat)
                    results.append(row)
                    raw_file.write(json.dumps(row) + "\n")
                    raw_file.flush()

    summary = summarize(results)
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_text_summary(summary_txt_path, summary)
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_json_path}")
    print(f"Wrote {summary_txt_path}")


if __name__ == "__main__":
    main()
