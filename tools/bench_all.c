// NPU Benchmark — all models (CLIP text, CLIP visual, YOLOv8n)
// Handles different input formats per model
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

static int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static void run_bench(void *lib, context_handler_t *ctx, const char *model, int N) {
    fn_load_graph p_load = dlsym(lib, "noe_load_graph");
    fn_create_job p_create = dlsym(lib, "noe_create_job");
    fn_get_tensor_desc p_tdesc = dlsym(lib, "noe_get_tensor_descriptor");
    fn_load_tensor p_ltensor = dlsym(lib, "noe_load_tensor");
    fn_infer_sync p_infer = dlsym(lib, "noe_job_infer_sync");
    fn_get_tensor p_gtensor = dlsym(lib, "noe_get_tensor");
    fn_clean_job p_clean = dlsym(lib, "noe_clean_job");
    fn_unload p_unload = dlsym(lib, "noe_unload_graph");

    uint64_t gid = 0;
    noe_status_t st = p_load(ctx, model, &gid, NULL);
    if (st != 0) { printf("  FAILED to load: %s (err=%u)\n\n", model, st); return; }

    tensor_desc_t in_desc, out_desc;
    p_tdesc(ctx, gid, 0, 0, &in_desc);
    p_tdesc(ctx, gid, 1, 0, &out_desc);

    printf("  Input: %u bytes (dtype=%u), Output: %u bytes (dtype=%u)\n",
           in_desc.size, in_desc.data_type, out_desc.size, out_desc.data_type);

    // Allocate and zero-fill input (works for any model)
    void *input = calloc(1, in_desc.size);
    void *output = malloc(out_desc.size);

    // Warmup
    for (int i = 0; i < 3; i++) {
        job_config_npu_t nc = {0};
        job_config_t c = { .conf_j_npu = &nc };
        uint64_t jid = 0;
        p_create(ctx, gid, &jid, &c);
        p_ltensor(ctx, jid, 0, input);
        p_infer(ctx, jid, 10000);
        p_clean(ctx, jid);
    }

    double *times = malloc(N * sizeof(double));
    double *full_times = malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        double t0_full = now_ms();
        job_config_npu_t nc = {0};
        job_config_t c = { .conf_j_npu = &nc };
        uint64_t jid = 0;
        p_create(ctx, gid, &jid, &c);
        p_ltensor(ctx, jid, 0, input);

        double t0 = now_ms();
        p_infer(ctx, jid, 10000);
        double t1 = now_ms();

        p_gtensor(ctx, jid, 1, 0, output);
        p_clean(ctx, jid);
        double t1_full = now_ms();

        times[i] = t1 - t0;
        full_times[i] = t1_full - t0_full;
    }

    qsort(times, N, sizeof(double), cmp_double);
    qsort(full_times, N, sizeof(double), cmp_double);

    double sum = 0, sum_f = 0;
    for (int i = 0; i < N; i++) { sum += times[i]; sum_f += full_times[i]; }
    double avg = sum / N, avg_f = sum_f / N;

    printf("  Inference only:  mean=%.2f ms  p50=%.2f  p95=%.2f  p99=%.2f  min=%.2f  max=%.2f  → %.1f inf/s\n",
           avg, times[N/2], times[(int)(N*0.95)], times[(int)(N*0.99)], times[0], times[N-1], 1000.0/avg);
    printf("  Full pipeline:   mean=%.2f ms  p50=%.2f  p95=%.2f  p99=%.2f  min=%.2f  max=%.2f  → %.1f inf/s\n",
           avg_f, full_times[N/2], full_times[(int)(N*0.95)], full_times[(int)(N*0.99)], full_times[0], full_times[N-1], 1000.0/avg_f);

    free(times); free(full_times); free(input); free(output);
    p_unload(ctx, gid);
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 100;

    void *lib = dlopen("/usr/share/cix/lib/libnoe.so", RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    fn_init p_init = dlsym(lib, "noe_init_context");
    context_handler_t *ctx = NULL;
    p_init(&ctx);

    const char *models[][2] = {
        {"/home/thomas/models/clip_txt.cix", "CLIP Text (ViT-B/32)"},
        {"/home/thomas/models/clip_visual.cix", "CLIP Visual (ViT-B/32)"},
        {"/home/thomas/models/yolov8n.cix", "YOLOv8n Detection"},
        {NULL, NULL}
    };

    printf("NPU Benchmark — %d iterations per model\n", N);
    printf("Kernel: "); fflush(stdout);
    system("uname -r");
    printf("NPU: Zhouyi Z3 (3 cores × 4 TECs)\n\n");

    for (int i = 0; models[i][0]; i++) {
        printf("--- %s ---\n", models[i][1]);
        run_bench(lib, ctx, models[i][0], N);
        printf("\n");
    }

    fn_deinit p_deinit = dlsym(lib, "noe_deinit_context");
    p_deinit(ctx);
    return 0;
}
