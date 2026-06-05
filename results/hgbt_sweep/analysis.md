# HGBT Scaling Sweep Analysis

Run window: 2026-06-05T16:45:58Z to about 2026-06-05T17:06:00Z.

GC was disabled. Each completed point has 12 repeats, except
`huge_leaf1000` at 64 threads, which has 6 repeats because the sweep was
stopped at the requested time cap.

## Completed Workloads

- `baseline_clf`: 100k samples, 100 features, max_features=0.5, max_leaf_nodes=31, max_iter=100.
- `low_features_per_split`: same as baseline but max_features=0.1.
- `all_features_per_split`: same as baseline but max_features=1.0.
- `stumps_leaf2`: same as baseline but max_leaf_nodes=2.
- `large_leaf255`: same base shape, max_leaf_nodes=255, max_iter=50.
- `huge_leaf1000`: 60k samples, 100 features, max_leaf_nodes=1000, max_iter=20.

Planned `few_features`, `many_features`, and `baseline_reg` workloads did not
start before the time cap.

## Main Findings

The earlier conclusion mostly applies to medium-depth trees: 16-32 threads is
the useful range, and 64 threads is usually slower.

For medium-depth 31-leaf trees:

- `baseline_clf` is best at 32 threads: 0.987 s, 8.91x speedup.
- `low_features_per_split` is best at 32 threads: 0.960 s, 7.91x speedup.
- `all_features_per_split` is best at 32 threads: 0.944 s, 9.64x speedup.
- 64 threads regresses in all three cases.

For shallow stumps:

- `stumps_leaf2` keeps improving up to 64 threads: 0.282 s, 8.81x speedup.
- This workload is dominated by binning and root histogram work, so it avoids
  the high-thread split-application penalties seen in larger trees.

For large trees:

- `large_leaf255` is best at 16 threads: 2.683 s, 3.55x speedup.
- `huge_leaf1000` is best at 16 threads: 2.917 s, 1.59x speedup.
- 32 and 64 threads are actively bad for large trees. In `huge_leaf1000`,
  32 threads is slower than 1 thread, and 64 threads is worse still.

## What Scales

Binning transform scales well:

- Baseline transform: 0.913 s at 1 thread, 0.031 s at 32, 0.019 s at 64.
- It is one of the few phases that still improves at 64 threads.

Bin fitting scales only to a moderate thread count:

- Baseline bin fit: 0.710 s at 1 thread, 0.186 s at 4, 0.113 s at 16,
  0.109 s at 32, 0.109 s at 64.
- Low-features-per-split bin fit: 0.702 s at 1 thread, 0.187 s at 4,
  0.111 s at 16, 0.108 s at 32, 0.112 s at 64.
- All-features-per-split bin fit: 0.711 s at 1 thread, 0.187 s at 4,
  0.102 s at 16, 0.106 s at 32, 0.122 s at 64.
- Large-leaf bin fit regresses at high thread count: 0.698 s at 1 thread,
  0.148 s at 16, 0.159 s at 32, 0.217 s at 64.

This is different from bin transform. In scikit-learn, bin fit computes
thresholds feature-by-feature through joblib's threading backend, while bin
transform calls the OpenMP `_map_to_bins` implementation. The fit stage gets a
large gain from 1 to 4 threads and a smaller gain up to around 16 threads, but
the work per feature is not large enough in these workloads to keep 32-64
threads efficiently occupied.

Histogram computation scales best for medium trees:

- Baseline histogram: 5.689 s at 1 thread, 0.393 s at 32.
- All-features histogram: 5.469 s at 1 thread, 0.387 s at 32.

## What Does Not Scale

Split finding and split application become the limiter.

Baseline apply split:

- 0.127 s at 4 threads
- 0.129 s at 32 threads
- 0.208 s at 64 threads

Large-leaf apply split:

- 0.318 s at 4 threads
- 0.453 s at 16 threads
- 0.742 s at 32 threads
- 1.176 s at 64 threads

Huge-leaf apply split:

- 0.322 s at 4 threads
- 0.654 s at 16 threads
- 1.357 s at 32 threads
- 2.029 s at 64 threads

That high-thread regression dominates large-tree workloads.

## Practical Thread Guidance

- Medium 31-leaf trees: test 16 and 32; prefer 32 only if it wins on the target machine.
- Large trees, hundreds of leaves or more: prefer 4-16; avoid 32/64 unless measured.
- Stumps or very shallow trees: 64 can still be useful because binning/root histogram work dominates.
- More features per split can improve scaling because there is more parallel work per node.
- Very large leaf counts expose split/partition overhead and scale much worse.
