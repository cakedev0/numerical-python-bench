# simple-expes

Small benchmark scripts for comparing numerical Python, scikit-learn, and OpenMP behavior.

## Pixi

Install the default environment:

```bash
pixi install
```

Run the scikit-learn benchmark:

```bash
pixi run python bench_sklearn.py --problem clf --threads 8
```

Run the HGBT scaling sweep:

```bash
pixi run python bench_hgbt_sweep.py --threads 1,4,16,32,64 --repeats 2
```

The saved sweep notes are in `results/hgbt_sweep/analysis.md`.

Run the Cython/OpenMP `prange` startup benchmark:

```bash
pixi run python bench_prange_startup.py --threads 0,1,2,4 --repeats 10000 --samples 3
```

Run with the Intel NumPy environment:

```bash
pixi run -e intel-numpy python bench_matmul.py --threads 8
```

## C/C++ OpenMP

Build:

```bash
gcc -O3 -fopenmp bench_omp_startup.c -o bench_omp_startup_gcc
g++ -O3 -fopenmp bench_omp_startup.c -o bench_omp_startup_gxx
```

Run:

```bash
./bench_omp_startup_gcc --threads 0,1,2,4 --repeats 10000 --samples 3
```
