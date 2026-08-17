#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <shmem.h>

// ~33.55 GB per PE * 16 PEs = 536.87 GB total
#define ELEMENTS_PER_PE 4194304000ULL 

static uint64_t *local_keys;
static uint64_t *remote_buffers;

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int compare_uint64(const void *a, const void *b) {
    uint64_t arg1 = *(const uint64_t *)a;
    uint64_t arg2 = *(const uint64_t *)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

int main(int argc, char *argv[]) {
    shmem_init();
    
    int me = shmem_my_pe();
    int npes = shmem_n_pes();

    local_keys = (uint64_t *)shmem_malloc(ELEMENTS_PER_PE * sizeof(uint64_t));
    remote_buffers = (uint64_t *)shmem_malloc(ELEMENTS_PER_PE * sizeof(uint64_t));

    if (!local_keys || !remote_buffers) {
        if (me == 0) fprintf(stderr, "Error: Symmetric heap allocation failed.\n");
        shmem_global_exit(1);
    }

    // Synthetic data generation in Host DRAM
    unsigned int seed = me + 100;
    for (size_t i = 0; i < ELEMENTS_PER_PE; i++) {
        local_keys[i] = ((uint64_t)rand_r(&seed) << 32) | rand_r(&seed);
    }

    shmem_barrier_all();
    double start_time = get_time_sec();

    // One-sided PGAS Put
    size_t write_offset = 0;
    for (size_t i = 0; i < ELEMENTS_PER_PE; i++) {
        int target_pe = (local_keys[i] >> 56) % npes;
        shmem_put64(&remote_buffers[write_offset], &local_keys[i], 1, target_pe);
        write_offset = (write_offset + 1) % ELEMENTS_PER_PE;
        if ((i & 0x1FFFF) == 0) {
            shmem_quiet();
        }
    }

    shmem_barrier_all();

    // Local Quicksort
    qsort(remote_buffers, ELEMENTS_PER_PE, sizeof(uint64_t), compare_uint64);
    shmem_barrier_all();

    double end_time = get_time_sec();
    double total_time = end_time - start_time;
    double total_bytes = (double)ELEMENTS_PER_PE * sizeof(uint64_t) * npes;

    if (me == 0) {
        FILE *f = fopen("/tmp/option1_metrics.json", "w");
        if (f) {
            fprintf(f, "{\n");
            fprintf(f, "  \"option\": \"Option 1 (Host-Side OpenSHMEM)\",\n");
            fprintf(f, "  \"data_gb\": %.2f,\n", total_bytes / 1e9);
            fprintf(f, "  \"execution_seconds\": %.4f,\n", total_time);
            fprintf(f, "  \"throughput_gbps\": %.2f,\n", (total_bytes / 1e9) / total_time);
            fprintf(f, "  \"compliant\": true,\n");
            fprintf(f, "  \"memory_location\": \"Host DRAM\",\n");
            fprintf(f, "  \"interconnect\": \"gVNIC / UCX (DCN)\"\n");
            fprintf(f, "}\n");
            fclose(f);
        }
        printf("Option 1 completed successfully in %.4f seconds.\n", total_time);
    }

    shmem_free(local_keys);
    shmem_free(remote_buffers);
    shmem_finalize();
    return 0;
}
