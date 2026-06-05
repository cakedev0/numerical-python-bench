# bench_matmul.py
import os, time, argparse
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=8000)
parser.add_argument("--threads", type=int, default=None)
parser.add_argument("--repeat", type=int, default=5)
args = parser.parse_args()

print("ENV:", {k: os.environ.get(k) for k in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "OMP_PROC_BIND", "OMP_PLACES"
]})
# np.show_config()
print(threadpool_info())

rng = np.random.default_rng(0)
A = rng.standard_normal((args.n, args.n), dtype=np.float64)
B = rng.standard_normal((args.n, args.n), dtype=np.float64)

ctx = threadpool_limits(args.threads) if args.threads else threadpool_limits()
with ctx:
    times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        C = A @ B
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"{dt:.3f}s")

print("median:", np.median(times))
