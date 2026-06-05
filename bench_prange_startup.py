#!/usr/bin/env python
"""Measure the startup cost of Cython/OpenMP prange loops.

The benchmark compiles a tiny Cython extension on first use. It measures an
empty prange loop repeated many times in C so Python call overhead is outside
the timed inner loop. Reported times are per prange invocation.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import statistics
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BUILD_DIR = HERE / "_cy_prange_startup_build"
MODULE_NAME = "_prange_startup_bench"


PYX_SOURCE = r"""
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from cython.parallel cimport parallel, prange

cdef extern from "omp.h":
    double omp_get_wtime() nogil


cpdef double bench_empty_loop(Py_ssize_t repeats):
    cdef Py_ssize_t r
    cdef double start

    with nogil:
        start = omp_get_wtime()
        for r in range(repeats):
            pass

        return omp_get_wtime() - start


cpdef double bench_prange(int n_threads, Py_ssize_t repeats, Py_ssize_t loop_iters):
    cdef Py_ssize_t r
    cdef Py_ssize_t i
    cdef double start

    with nogil:
        start = omp_get_wtime()
        for r in range(repeats):
            for i in prange(loop_iters, num_threads=n_threads, schedule="static"):
                pass

        return omp_get_wtime() - start


cpdef double bench_prange_hoisted(int n_threads, Py_ssize_t repeats, Py_ssize_t loop_iters):
    cdef Py_ssize_t r = 0
    cdef Py_ssize_t i = 0
    cdef double start

    with nogil:
        start = omp_get_wtime()
        with parallel(num_threads=n_threads):
            for r in range(repeats):
                for i in prange(loop_iters, schedule="static"):
                    pass

        return omp_get_wtime() - start


cpdef double bench_range(Py_ssize_t repeats, Py_ssize_t loop_iters):
    cdef Py_ssize_t r
    cdef Py_ssize_t i
    cdef volatile Py_ssize_t sink = 0
    cdef double start

    with nogil:
        start = omp_get_wtime()
        for r in range(repeats):
            for i in range(loop_iters):
                sink = i

        return omp_get_wtime() - start
"""


SETUP_SOURCE = r"""
from Cython.Build import cythonize
from setuptools import Extension, setup


extensions = [
    Extension(
        "_prange_startup_bench",
        ["_prange_startup_bench.pyx"],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]


setup(
    name="_prange_startup_bench",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
"""


def parse_threads(spec: str, max_threads: int) -> list[int]:
    if spec == "all":
        return list(range(0, max_threads + 1))

    values: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, stop_s = chunk.split("-", 1)
            values.update(range(int(start_s), int(stop_s) + 1))
        else:
            values.add(int(chunk))

    threads = sorted(t for t in values if 0 <= t <= max_threads)
    if not threads:
        raise ValueError("thread list is empty after applying max_threads")
    return threads


def default_thread_spec(max_threads: int) -> str:
    values = {0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, max_threads}
    return ",".join(str(t) for t in sorted(v for v in values if v <= max_threads))


def build_extension(force: bool = False) -> Path:
    BUILD_DIR.mkdir(exist_ok=True)
    pyx = BUILD_DIR / f"{MODULE_NAME}.pyx"
    setup_py = BUILD_DIR / "setup.py"
    rebuild_needed = force

    if (pyx.read_text() if pyx.exists() else "") != PYX_SOURCE:
        pyx.write_text(PYX_SOURCE)
        rebuild_needed = True
    if (setup_py.read_text() if setup_py.exists() else "") != SETUP_SOURCE:
        setup_py.write_text(SETUP_SOURCE)
        rebuild_needed = True

    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    built = next(BUILD_DIR.glob(f"{MODULE_NAME}*{suffix}"), None)
    if built is not None and not rebuild_needed:
        return built

    build_cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    if force:
        build_cmd.append("--force")
    subprocess.run(build_cmd, cwd=BUILD_DIR, check=True)

    built = next(BUILD_DIR.glob(f"{MODULE_NAME}*{suffix}"), None)
    if built is None:
        raise RuntimeError("Cython extension build did not produce a shared object")
    return built


def load_extension(force_rebuild: bool):
    try:
        import Cython  # noqa: F401
        import setuptools  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing build dependency. Install with `pixi install` or "
            "`python -m pip install cython setuptools`."
        ) from exc

    module_path = build_extension(force=force_rebuild)
    spec = importlib.util.spec_from_file_location(MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import extension from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(samples: list[float], repeats: int, baseline_s: float = 0.0) -> dict[str, float]:
    adjusted = [(sample - baseline_s) / repeats for sample in samples]
    return {
        "min_us": min(adjusted) * 1e6,
        "median_us": statistics.median(adjusted) * 1e6,
        "mean_us": statistics.fmean(adjusted) * 1e6,
        "max_us": max(adjusted) * 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-threads", type=int, default=172)
    parser.add_argument(
        "--threads",
        default=None,
        help="'all', '0,1,2,4', or ranges like '0-16,32,64'. 0 means plain range.",
    )
    parser.add_argument("--repeats", type=int, default=20_000)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--loop-iters",
        type=int,
        default=1,
        help="Iterations in each prange loop. The default matches `prange(1)`.",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    if args.max_threads < 1:
        raise SystemExit("--max-threads must be >= 1")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    if args.loop_iters < 0:
        raise SystemExit("--loop-iters must be >= 0")

    thread_spec = args.threads or default_thread_spec(args.max_threads)
    threads = parse_threads(thread_spec, args.max_threads)
    mod = load_extension(args.force_rebuild)

    env = {k: os.environ.get(k, "") for k in ["OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES"]}
    print(
        "env: "
        f"OMP_NUM_THREADS={env['OMP_NUM_THREADS']} "
        f"OMP_PROC_BIND={env['OMP_PROC_BIND']} "
        f"OMP_PLACES={env['OMP_PLACES']}"
    )
    print(f"repeats={args.repeats} samples={args.samples} loop_iters={args.loop_iters}")
    print()
    print(
        f"{'threads':>8} {'mode':<14} "
        f"{'min_us':>12} {'median_us':>12} {'mean_us':>12} {'max_us':>12}"
    )
    print(
        f"{'-------':>8} {'----':<14} "
        f"{'------':>12} {'---------':>12} {'-------':>12} {'------':>12}"
    )

    baseline_samples = [mod.bench_empty_loop(args.repeats) for _ in range(args.samples)]
    baseline_s = statistics.median(baseline_samples)

    for n_threads in threads:
        if n_threads == 0:
            samples = [mod.bench_range(args.repeats, args.loop_iters) for _ in range(args.samples)]
            row = summarize(samples, args.repeats)
            print(
                f"{n_threads:8d} {'range':<14} "
                f"{row['min_us']:12.3f} "
                f"{row['median_us']:12.3f} "
                f"{row['mean_us']:12.3f} "
                f"{row['max_us']:12.3f}"
            )
            continue

        samples = [
            mod.bench_prange(n_threads, args.repeats, args.loop_iters)
            for _ in range(args.samples)
        ]
        row = summarize(samples, args.repeats, baseline_s)
        print(
            f"{n_threads:8d} {'parallel_for':<14} "
            f"{row['min_us']:12.3f} "
            f"{row['median_us']:12.3f} "
            f"{row['mean_us']:12.3f} "
            f"{row['max_us']:12.3f}"
        )

        samples = [
            mod.bench_prange_hoisted(n_threads, args.repeats, args.loop_iters)
            for _ in range(args.samples)
        ]
        row = summarize(samples, args.repeats, baseline_s)
        print(
            f"{n_threads:8d} {'hoisted_for':<14} "
            f"{row['min_us']:12.3f} "
            f"{row['median_us']:12.3f} "
            f"{row['mean_us']:12.3f} "
            f"{row['max_us']:12.3f}"
        )


if __name__ == "__main__":
    main()
