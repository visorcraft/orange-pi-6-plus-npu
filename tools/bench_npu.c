// NPU Benchmark — CLIP text encoder throughput
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <time.h>
#include <math.h>

typedef struct { uint32_t handle; } context_handler_t;
typedef uint32_t noe_status_t;
typedef struct { uint32_t id; uint32_t size; float scale; int32_t zero_point; uint32_t data_type; } tensor_desc_t;
typedef struct { uint32_t misc; void *fm_idxes; int32_t fm_idxes_cnt; void *dynshape; } job_config_npu_t;
typedef struct { job_config_npu_t *conf_j_npu; } job_config_t;

typedef noe_status_t (*fn_init)(context_handler_t**);
typedef noe_status_t (*fn_load_graph)(const context_handler_t*, const char*, uint64_t*, void*);
typedef noe_status_t (*fn_create_job)(const context_handler_t*, uint64_t, uint64_t*, job_config_t*);
typedef noe_status_t (*fn_get_tensor_desc)(const context_handler_t*, uint64_t, int, uint32_t, tensor_desc_t*);
typedef noe_status_t (*fn_load_tensor)(const context_handler_t*, uint64_t, uint32_t, const void*);
typedef noe_status_t (*fn_infer_sync)(const context_handler_t*, uint64_t, int32_t);
typedef noe_status_t (*fn_get_tensor)(const context_handler_t*, uint64_t, int, uint32_t, void*);
typedef noe_status_t (*fn_clean_job)(const context_handler_t*, uint64_t);
typedef noe_status_t (*fn_unload)(const context_handler_t*, uint64_t);
typedef noe_status_t (*fn_deinit)(const context_handler_t*);

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 100;
    const char *model = argc > 2 ? argv[2] : "clip_txt.cix";

    void *lib = dlopen("/usr/share/cix/lib/libnoe.so", RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    fn_init p_init = dlsym(lib, "noe_init_context");
    fn_load_graph p_load = dlsym(lib, "noe_load_graph");
    fn_create_job p_create = dlsym(lib, "noe_create_job");
    fn_get_tensor_desc p_tdesc = dlsym(lib, "noe_get_tensor_descriptor");
    fn_load_tensor p_ltensor = dlsym(lib, "noe_load_tensor");
    fn_infer_sync p_infer = dlsym(lib, "noe_job_infer_sync");
    fn_get_tensor p_gtensor = dlsym(lib, "noe_get_tensor");
    fn_clean_job p_clean = dlsym(lib, "noe_clean_job");
    fn_unload p_unload = dlsym(lib, "noe_unload_graph");
    fn_deinit p_deinit = dlsym(lib, "noe_deinit_context");

    context_handler_t *ctx = NULL;
    p_init(&ctx);
    uint64_t gid = 0;
    p_load(ctx, model, &gid, NULL);

    tensor_desc_t in_desc, out_desc;
    p_tdesc(ctx, gid, 0, 0, &in_desc);
    p_tdesc(ctx, gid, 1, 0, &out_desc);

    int out_dims = out_desc.size / 2; // INT16
    printf("Model: %s\n", model);
    printf("Input: %u bytes (dtype=%u), Output: %u bytes (%d dims, dtype=%u)\n",
           in_desc.size, in_desc.data_type, out_desc.size, out_dims, out_desc.data_type);
    printf("Running %d iterations...\n\n", N);

    // Prepare input
    int32_t tokens[77] = {0};
    tokens[0] = 49406; tokens[1] = 3306; tokens[2] = 1002; tokens[3] = 49407;

    // Warmup (3 iterations)
    for (int i = 0; i < 3; i++) {
        job_config_npu_t nc = {0};
        job_config_t c = { .conf_j_npu = &nc };
        uint64_t jid = 0;
        p_create(ctx, gid, &jid, &c);
        p_ltensor(ctx, jid, 0, tokens);
        p_infer(ctx, jid, 10000);
        p_clean(ctx, jid);
    }

    // Benchmark individual iterations
    double *times = malloc(N * sizeof(double));
    double *full_times = malloc(N * sizeof(double)); // including job create/clean
    void *outbuf = malloc(out_desc.size);

    for (int i = 0; i < N; i++) {
        double t0_full = now_ms();

        job_config_npu_t nc = {0};
        job_config_t c = { .conf_j_npu = &nc };
        uint64_t jid = 0;
        p_create(ctx, gid, &jid, &c);
        p_ltensor(ctx, jid, 0, tokens);

        double t0 = now_ms();
        p_infer(ctx, jid, 10000);
        double t1 = now_ms();

        p_gtensor(ctx, jid, 1, 0, outbuf);
        p_clean(ctx, jid);

        double t1_full = now_ms();
        times[i] = t1 - t0;
        full_times[i] = t1_full - t0_full;
    }

    // Stats
    double sum = 0, sum_full = 0, min_t = 1e9, max_t = 0, min_f = 1e9, max_f = 0;
    for (int i = 0; i < N; i++) {
        sum += times[i];
        sum_full += full_times[i];
        if (times[i] < min_t) min_t = times[i];
        if (times[i] > max_t) max_t = times[i];
        if (full_times[i] < min_f) min_f = full_times[i];
        if (full_times[i] > max_f) max_f = full_times[i];
    }
    double avg = sum / N;
    double avg_full = sum_full / N;

    // Stddev
    double var = 0, var_f = 0;
    for (int i = 0; i < N; i++) {
        var += (times[i] - avg) * (times[i] - avg);
        var_f += (full_times[i] - avg_full) * (full_times[i] - avg_full);
    }
    double stddev = sqrt(var / N);
    double stddev_f = sqrt(var_f / N);

    // P50, P95, P99
    // Sort times for percentiles
    for (int i = 0; i < N - 1; i++)
        for (int j = i + 1; j < N; j++)
            if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }

    printf("=== NPU Inference Only (noe_job_infer_sync) ===\n");
    printf("  Mean:   %.2f ms\n", avg);
    printf("  Stddev: %.2f ms\n", stddev);
    printf("  Min:    %.2f ms\n", min_t);
    printf("  Max:    %.2f ms\n", max_t);
    printf("  P50:    %.2f ms\n", times[N/2]);
    printf("  P95:    %.2f ms\n", times[(int)(N*0.95)]);
    printf("  P99:    %.2f ms\n", times[(int)(N*0.99)]);
    printf("  Throughput: %.1f inferences/sec\n", 1000.0 / avg);
    printf("\n");

    printf("=== Full Pipeline (create_job + load + infer + read + clean) ===\n");
    printf("  Mean:   %.2f ms\n", avg_full);
    printf("  Stddev: %.2f ms\n", stddev_f);
    printf("  Min:    %.2f ms\n", min_f);
    printf("  Max:    %.2f ms\n", max_f);
    printf("  Throughput: %.1f inferences/sec\n", 1000.0 / avg_full);
    printf("\n");

    // Model-specific metrics
    // CLIP ViT-B/32 text: ~63M params, ~243MB unquantized (FP32)
    // Quantized to mixed INT8/INT16 = ~71MB on disk
    // Operations: 12 transformer layers, each with attention + FFN
    // Rough FLOP estimate for CLIP text: ~1.2 GFLOPS per inference
    double gflops_est = 1.2; // rough estimate for CLIP ViT-B/32 text encoder
    printf("=== Derived Metrics ===\n");
    printf("  Model params: ~63M\n");
    printf("  Model size (quantized): 71 MB\n");
    printf("  Output dims: %d\n", out_dims);
    printf("  Est. compute: ~%.1f GFLOPS/inference\n", gflops_est);
    printf("  Est. NPU throughput: ~%.1f GFLOPS (at %.1f inf/s)\n",
           gflops_est * (1000.0 / avg), 1000.0 / avg);
    printf("  Embedding throughput: %.0f dims/ms (%.0f dims/s)\n",
           out_dims / avg, out_dims * 1000.0 / avg);
    printf("  Data throughput: %.2f MB/s output\n",
           (out_desc.size / 1024.0 / 1024.0) * (1000.0 / avg));

    free(times);
    free(full_times);
    free(outbuf);
    p_unload(ctx, gid);
    p_deinit(ctx);
    return 0;
}
