// Measure OpenMP parallel-for startup cost directly from C/C++.
//
// Build:
//   gcc -O3 -fopenmp bench_omp_startup.c -o bench_omp_startup_gcc
//   g++ -O3 -fopenmp bench_omp_startup.c -o bench_omp_startup_gxx
//
// Run:
//   ./bench_omp_startup_gcc --threads 0,1,2,4 --repeats 100000 --samples 7
//   ./bench_omp_startup_gxx --threads all --max-threads 172
//
// Output:
//   mode=parallel_for repeatedly enters/leaves an OpenMP team.
//   mode=hoisted_for enters one team and repeatedly executes omp for inside it.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef long long i64;

typedef struct {
    int max_threads;
    char threads_spec[1024];
    i64 repeats;
    int samples;
    i64 loop_iters;
} Options;

typedef struct {
    int *data;
    int size;
    int capacity;
} IntVec;

static void usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s [--max-threads N] [--threads SPEC] [--repeats N] "
        "[--samples N] [--loop-iters N]\n\n"
        "SPEC can be 'all', '0,1,2,4', or ranges like '0-16,32,64'.\n"
        "Thread count 0 means a plain serial range-style loop.\n",
        program);
}

static i64 parse_i64(const char *s, const char *name) {
    char *end = NULL;
    i64 value = strtoll(s, &end, 10);
    if (end == s || *end != '\0') {
        fprintf(stderr, "invalid %s: %s\n", name, s);
        exit(2);
    }
    return value;
}

static void init_options(Options *opts) {
    opts->max_threads = 172;
    opts->threads_spec[0] = '\0';
    opts->repeats = 20000;
    opts->samples = 7;
    opts->loop_iters = 1;
}

static void parse_args(int argc, char **argv, Options *opts) {
    int i;
    init_options(opts);

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "--max-threads") == 0 && i + 1 < argc) {
            opts->max_threads = (int)parse_i64(argv[++i], "--max-threads");
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            snprintf(opts->threads_spec, sizeof(opts->threads_spec), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            opts->repeats = parse_i64(argv[++i], "--repeats");
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            opts->samples = (int)parse_i64(argv[++i], "--samples");
        } else if (strcmp(argv[i], "--loop-iters") == 0 && i + 1 < argc) {
            opts->loop_iters = parse_i64(argv[++i], "--loop-iters");
        } else {
            usage(argv[0]);
            exit(2);
        }
    }

    if (opts->max_threads < 1) {
        fprintf(stderr, "--max-threads must be >= 1\n");
        exit(2);
    }
    if (opts->repeats < 1) {
        fprintf(stderr, "--repeats must be >= 1\n");
        exit(2);
    }
    if (opts->samples < 1) {
        fprintf(stderr, "--samples must be >= 1\n");
        exit(2);
    }
    if (opts->loop_iters < 0) {
        fprintf(stderr, "--loop-iters must be >= 0\n");
        exit(2);
    }
}

static void int_vec_push(IntVec *vec, int value) {
    int new_capacity;
    int *new_data;

    if (vec->size == vec->capacity) {
        new_capacity = vec->capacity == 0 ? 16 : vec->capacity * 2;
        new_data = (int *)realloc(vec->data, (size_t)new_capacity * sizeof(int));
        if (new_data == NULL) {
            fprintf(stderr, "out of memory\n");
            exit(1);
        }
        vec->data = new_data;
        vec->capacity = new_capacity;
    }
    vec->data[vec->size++] = value;
}

static int int_cmp(const void *a, const void *b) {
    int lhs = *(const int *)a;
    int rhs = *(const int *)b;
    return (lhs > rhs) - (lhs < rhs);
}

static int double_cmp(const void *a, const void *b) {
    double lhs = *(const double *)a;
    double rhs = *(const double *)b;
    return (lhs > rhs) - (lhs < rhs);
}

static void add_thread_value(IntVec *threads, int value, int max_threads) {
    if (0 <= value && value <= max_threads) {
        int_vec_push(threads, value);
    }
}

static void add_default_threads(IntVec *threads, int max_threads) {
    int values[] = {0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, max_threads};
    int n_values = (int)(sizeof(values) / sizeof(values[0]));
    int i;

    for (i = 0; i < n_values; i++) {
        add_thread_value(threads, values[i], max_threads);
    }
}

static void parse_thread_spec(const char *spec, int max_threads, IntVec *threads) {
    char *copy;
    char *chunk;

    if (spec[0] == '\0') {
        add_default_threads(threads, max_threads);
    } else if (strcmp(spec, "all") == 0) {
        int t;
        for (t = 0; t <= max_threads; t++) {
            int_vec_push(threads, t);
        }
    } else {
        copy = (char *)malloc(strlen(spec) + 1);
        if (copy == NULL) {
            fprintf(stderr, "out of memory\n");
            exit(1);
        }
        strcpy(copy, spec);

        chunk = strtok(copy, ",");
        while (chunk != NULL) {
            char *dash = strchr(chunk, '-');
            if (dash != NULL) {
                int start;
                int stop;
                *dash = '\0';
                start = (int)parse_i64(chunk, "thread range start");
                stop = (int)parse_i64(dash + 1, "thread range stop");
                if (start <= stop) {
                    int t;
                    for (t = start; t <= stop; t++) {
                        add_thread_value(threads, t, max_threads);
                    }
                }
            } else {
                add_thread_value(threads, (int)parse_i64(chunk, "thread value"), max_threads);
            }
            chunk = strtok(NULL, ",");
        }
        free(copy);
    }

    if (threads->size == 0) {
        fprintf(stderr, "thread list is empty after applying max_threads\n");
        exit(2);
    }

    qsort(threads->data, (size_t)threads->size, sizeof(int), int_cmp);
    {
        int read;
        int write = 0;
        for (read = 0; read < threads->size; read++) {
            if (write == 0 || threads->data[read] != threads->data[write - 1]) {
                threads->data[write++] = threads->data[read];
            }
        }
        threads->size = write;
    }
}

static double bench_empty_loop(i64 repeats) {
    i64 r;
    double start = omp_get_wtime();
    for (r = 0; r < repeats; r++) {
    }
    return omp_get_wtime() - start;
}

static double bench_range(i64 repeats, i64 loop_iters) {
    i64 r;
    i64 i;
    volatile i64 sink = 0;
    double start = omp_get_wtime();

    for (r = 0; r < repeats; r++) {
        for (i = 0; i < loop_iters; i++) {
            sink = i;
        }
    }

    return omp_get_wtime() - start;
}

static double bench_omp_for(int n_threads, i64 repeats, i64 loop_iters) {
    i64 r;
    double start = omp_get_wtime();

    for (r = 0; r < repeats; r++) {
        i64 i;
#pragma omp parallel num_threads(n_threads)
        {
#pragma omp for schedule(static)
            for (i = 0; i < loop_iters; i++) {
                volatile i64 sink = i;
                (void)sink;
            }
        }
    }

    return omp_get_wtime() - start;
}

static double bench_omp_for_hoisted(int n_threads, i64 repeats, i64 loop_iters) {
    double start = omp_get_wtime();

#pragma omp parallel num_threads(n_threads)
    {
        i64 r;
        for (r = 0; r < repeats; r++) {
            i64 i;
#pragma omp for schedule(static)
            for (i = 0; i < loop_iters; i++) {
                volatile i64 sink = i;
                (void)sink;
            }
        }
    }

    return omp_get_wtime() - start;
}

static double median(double *values, int n) {
    qsort(values, (size_t)n, sizeof(double), double_cmp);
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

static void summarize(
    const double *samples,
    int n_samples,
    i64 repeats,
    double baseline_s,
    double *min_us,
    double *median_us,
    double *mean_us,
    double *max_us) {
    double *adjusted = (double *)malloc((size_t)n_samples * sizeof(double));
    double sum = 0.0;
    int i;

    if (adjusted == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    for (i = 0; i < n_samples; i++) {
        adjusted[i] = (samples[i] - baseline_s) / (double)repeats;
        sum += adjusted[i];
    }

    *min_us = adjusted[0];
    *max_us = adjusted[0];
    for (i = 1; i < n_samples; i++) {
        if (adjusted[i] < *min_us) {
            *min_us = adjusted[i];
        }
        if (adjusted[i] > *max_us) {
            *max_us = adjusted[i];
        }
    }
    *median_us = median(adjusted, n_samples);
    *mean_us = sum / (double)n_samples;

    *min_us *= 1e6;
    *median_us *= 1e6;
    *mean_us *= 1e6;
    *max_us *= 1e6;

    free(adjusted);
}

int main(int argc, char **argv) {
    Options opts;
    IntVec threads = {NULL, 0, 0};
    double *baseline_samples;
    double *samples;
    double baseline_s;
    int sample;
    int idx;

    parse_args(argc, argv, &opts);
    parse_thread_spec(opts.threads_spec, opts.max_threads, &threads);

    baseline_samples = (double *)malloc((size_t)opts.samples * sizeof(double));
    samples = (double *)malloc((size_t)opts.samples * sizeof(double));
    if (baseline_samples == NULL || samples == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    printf(
        "env: OMP_NUM_THREADS=%s OMP_PROC_BIND=%s OMP_PLACES=%s\n",
        getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "",
        getenv("OMP_PROC_BIND") ? getenv("OMP_PROC_BIND") : "",
        getenv("OMP_PLACES") ? getenv("OMP_PLACES") : "");
    printf(
        "repeats=%lld samples=%d loop_iters=%lld\n",
        opts.repeats,
        opts.samples,
        opts.loop_iters);
    printf(
        "\n%8s %-14s %12s %12s %12s %12s\n",
        "threads",
        "mode",
        "min_us",
        "median_us",
        "mean_us",
        "max_us");
    printf(
        "%8s %-14s %12s %12s %12s %12s\n",
        "-------",
        "----",
        "------",
        "---------",
        "-------",
        "------");

    for (sample = 0; sample < opts.samples; sample++) {
        baseline_samples[sample] = bench_empty_loop(opts.repeats);
    }
    baseline_s = median(baseline_samples, opts.samples);

    for (idx = 0; idx < threads.size; idx++) {
        int n_threads = threads.data[idx];
        double min_us;
        double median_us;
        double mean_us;
        double max_us;

        if (n_threads == 0) {
            for (sample = 0; sample < opts.samples; sample++) {
                samples[sample] = bench_range(opts.repeats, opts.loop_iters);
            }
            summarize(
                samples,
                opts.samples,
                opts.repeats,
                0.0,
                &min_us,
                &median_us,
                &mean_us,
                &max_us);
            printf(
                "%8d %-14s %12.3f %12.3f %12.3f %12.3f\n",
                n_threads,
                "range",
                min_us,
                median_us,
                mean_us,
                max_us);
            fflush(stdout);
            continue;
        }

        for (sample = 0; sample < opts.samples; sample++) {
            samples[sample] = bench_omp_for(n_threads, opts.repeats, opts.loop_iters);
        }
        summarize(
            samples,
            opts.samples,
            opts.repeats,
            baseline_s,
            &min_us,
            &median_us,
            &mean_us,
            &max_us);
        printf(
            "%8d %-14s %12.3f %12.3f %12.3f %12.3f\n",
            n_threads,
            "parallel_for",
            min_us,
            median_us,
            mean_us,
            max_us);
        fflush(stdout);

        for (sample = 0; sample < opts.samples; sample++) {
            samples[sample] = bench_omp_for_hoisted(n_threads, opts.repeats, opts.loop_iters);
        }
        summarize(
            samples,
            opts.samples,
            opts.repeats,
            baseline_s,
            &min_us,
            &median_us,
            &mean_us,
            &max_us);
        printf(
            "%8d %-14s %12.3f %12.3f %12.3f %12.3f\n",
            n_threads,
            "hoisted_for",
            min_us,
            median_us,
            mean_us,
            max_us);
        fflush(stdout);
    }

    free(samples);
    free(baseline_samples);
    free(threads.data);
    return 0;
}
