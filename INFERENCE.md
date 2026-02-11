# Running Models on the Zhouyi Z3 NPU — Orange Pi 6 Plus

*Last updated: 2026-02-10*
*Status: 3 models running — CLIP text/visual encoders + YOLOv8n object detection*
*Tested on: Armbian 26.2.0-trunk, kernel 6.18.9-current-arm64*

This guide covers running pre-compiled neural network models on the Zhouyi Z3 NPU using the NOE (Neural One Engine) C API. It assumes you've already built and loaded the `aipu.ko` kernel driver per [README.md](README.md).

---

## Table of Contents

1. [Overview](#overview)
2. [CIX AI Model Hub](#cix-ai-model-hub)
3. [The NOE C API](#the-noe-c-api)
4. [Critical ABI Gotcha](#critical-abi-gotcha)
5. [CLIP Text Embeddings — Full Example](#clip-text-embeddings--full-example)
6. [CLIP Visual Embeddings](#clip-visual-embeddings)
7. [YOLOv8n Object Detection](#yolov8n-object-detection)
8. [Performance Numbers](#performance-numbers)
9. [Building an Embedding Server](#building-an-embedding-server)
10. [Quantization & Dequantization](#quantization--dequantization)
11. [Alternative Paths](#alternative-paths)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The Zhouyi Z3 NPU runs pre-compiled model binaries (`.cix` files). The pipeline is:

```
ONNX model → CIX compiler (cixbuild) → .cix binary → libnoe.so → /dev/aipu → NPU hardware
```

You generally don't need to compile models yourself — CIX provides a model hub with pre-compiled binaries for common architectures. If you do need custom models, the `cixbuild` tool (part of the Compass toolchain) runs on x86 Linux.

**Supported model types:** CNNs (ResNet, MobileNet, YOLO, etc.), CLIP, vision transformers, and some language models. The NPU excels at INT8/INT16 quantized inference with static input shapes.

## CIX AI Model Hub

CIX maintains a model zoo with pre-compiled `.cix` binaries ready to run:

**Repository:** [modelscope.cn/cix/ai_model_hub_25_Q3](https://modelscope.cn/models/cix/ai_model_hub_25_Q3)

The repo uses git LFS for large files. The simplest way to download individual models:

```bash
# Direct download (no git-lfs needed):
curl -sL -o model.cix \
  "https://modelscope.cn/models/cix/ai_model_hub_25_Q3/resolve/master/models/<path>/model.cix"
```

### Repository Structure

```
models/
├── Audio/                    # Speech models
├── ComputeVision/            # Classification, detection, segmentation
├── Generative_AI/
│   └── Image_to_Text/
│       ├── onnx_clip/        # CLIP ViT-B/32 (text + visual encoders)
│       └── onnx_Chinese_clip/
└── MultiModal/               # Multi-modal models
```

Each model directory contains:
- `*.cix` — Pre-compiled NPU binary (the only file you need to run inference)
- `cfg/*.cfg` — Build config used by `cixbuild` (useful reference for quantization settings)
- `inference_npu.py` — Python inference script (reference implementation)
- `inference_onnx.py` — CPU-only ONNX inference (for comparison/validation)
- `model/*.onnx` — Original ONNX model (source, not needed for NPU inference)
- `datasets/` — Calibration data used during quantization

### Build Config Reference

The `.cfg` files reveal how models were quantized. Example from CLIP text encoder:

```ini
[Parser]
model_type = onnx
input_model = model/clip_text_model_vitb32.onnx
input = TEXT
input_shape = [1, 77]      # Static shape: batch=1, sequence_length=77

[Optimizer]
weight_bits = 8             # INT8 weights (with INT16 for specific layers)
activation_bits = 8         # INT8 activations
calibration_data = datasets/text_cal.npy
quantize_method_for_activation = per_tensor_asymmetric

[GBuilder]
target = X2_1204MP3         # Zhouyi Z3 target (3 cores, 4 TECs each)
outputs = clip_txt.cix
```

Key settings:
- `target = X2_1204MP3` — This is the Zhouyi Z3 on the CIX CD8180. Use this for any model you compile yourself.
- Mixed precision: Most layers INT8, attention projection layers INT16 for accuracy.
- Static input shapes are **mandatory** — the NPU cannot handle dynamic dimensions.

## The NOE C API

`libnoe.so` is the userspace library that communicates with the NPU via `/dev/aipu`. The full header is at `/usr/share/cix/include/npu/cix_noe_standard_api.h`.

### Core Workflow

```c
#include "cix_noe_standard_api.h"

// 1. Initialize context (opens /dev/aipu)
context_handler_t *ctx;
noe_init_context(&ctx);

// 2. Load compiled model
uint64_t graph_id;
noe_load_graph(ctx, "model.cix", &graph_id, NULL);

// 3. Query tensor shapes
tensor_desc_t in_desc, out_desc;
noe_get_tensor_descriptor(ctx, graph_id, NOE_TENSOR_TYPE_INPUT, 0, &in_desc);
noe_get_tensor_descriptor(ctx, graph_id, NOE_TENSOR_TYPE_OUTPUT, 0, &out_desc);

// 4. Create execution job (⚠️ see gotcha below)
uint64_t job_id;
job_config_npu_t npu_cfg = {0};
job_config_t cfg = { .conf_j_npu = &npu_cfg };
noe_create_job(ctx, graph_id, &job_id, &cfg);

// 5. Load input data
noe_load_tensor(ctx, job_id, 0, input_data);

// 6. Run inference (blocking)
noe_job_infer_sync(ctx, job_id, 10000);  // 10s timeout

// 7. Read output
void *output = malloc(out_desc.size);
noe_get_tensor(ctx, job_id, NOE_TENSOR_TYPE_OUTPUT, 0, output);

// 8. Cleanup
noe_clean_job(ctx, job_id);
noe_unload_graph(ctx, graph_id);
noe_deinit_context(ctx);
```

### Key Types

```c
typedef struct {
    uint32_t id;
    uint32_t size;          // Total bytes
    float    scale;         // Quantization scale
    int32_t  zero_point;    // Quantization zero point
    noe_data_type_t data_type;  // S8=1, S16=3, S32=7, F16=0xa, F32=0xb
} tensor_desc_t;
```

## Critical ABI Gotcha

**⚠️ `noe_create_job()` will SEGFAULT if you pass NULL as the config parameter.**

The C++ header declares a default parameter:
```cpp
noe_status_t noe_create_job(ctx, graph, &job, job_config_t *config = nullptr);
```

But the actual implementation **dereferences the config pointer unconditionally**:
```asm
; At offset +0x5c in noe_create_job:
ldr x3, [x21]    ; x21 = config pointer — dereferences it!
```

**The fix:** Always pass a valid, zero-initialized config struct:

```c
// WRONG — segfaults:
noe_create_job(ctx, graph_id, &job_id, NULL);

// CORRECT:
job_config_npu_t npu_cfg = {0};
job_config_t cfg = { .conf_j_npu = &npu_cfg };
noe_create_job(ctx, graph_id, &job_id, &cfg);
```

This applies to both C and C++ callers. The default parameter in the header is misleading.

## CLIP Text Embeddings — Full Example

This example runs CLIP ViT-B/32's text encoder on the NPU to generate 256-dimensional text embeddings.

### Download the Model

```bash
mkdir -p ~/models
curl -sL -o ~/models/clip_txt.cix \
  "https://modelscope.cn/models/cix/ai_model_hub_25_Q3/resolve/master/models/Generative_AI/Image_to_Text/onnx_clip/clip_txt.cix"
ls -lh ~/models/clip_txt.cix
# Expected: ~71MB
```

### Model Specs

| Property | Value |
|----------|-------|
| Architecture | CLIP ViT-B/32 text encoder |
| Parameters | ~63M |
| Model size | 71MB (INT8/INT16 quantized) |
| Input | `[1, 77]` INT32 — CLIP BPE token IDs |
| Output | `[1, 256]` INT16 — text embeddings (quantized) |
| Input scale | 1.0 (pass raw token IDs) |
| Output scale | ~27.67, zero_point=60 |
| Inference time | ~18ms |

### Minimal C Test Program

```c
// test_clip_inference.c
// Compile: gcc -o test_clip test_clip_inference.c -ldl
// Run: LD_LIBRARY_PATH=/usr/share/cix/lib ./test_clip

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <time.h>

// Minimal type definitions matching libnoe ABI
typedef struct { uint32_t handle; } context_handler_t;
typedef uint32_t noe_status_t;
typedef struct {
    uint32_t id; uint32_t size; float scale;
    int32_t zero_point; uint32_t data_type;
} tensor_desc_t;

typedef struct {
    uint32_t misc;
    void *fm_idxes;
    int32_t fm_idxes_cnt;
    void *dynshape;
} job_config_npu_t;

typedef struct {
    job_config_npu_t *conf_j_npu;
} job_config_t;

// Function pointer typedefs
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

int main(int argc, char *argv[]) {
    const char *model_path = argc > 1 ? argv[1] : "clip_txt.cix";

    void *lib = dlopen("libnoe.so", RTLD_NOW);
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

    // Initialize NPU
    context_handler_t *ctx = NULL;
    p_init(&ctx);
    printf("NPU context initialized\n");

    // Load model
    uint64_t gid = 0;
    noe_status_t ret = p_load(ctx, model_path, &gid, NULL);
    if (ret != 0) { fprintf(stderr, "load_graph failed: 0x%x\n", ret); return 1; }

    // Query tensor info
    tensor_desc_t in_desc, out_desc;
    p_tdesc(ctx, gid, 0, 0, &in_desc);  // 0 = NOE_TENSOR_TYPE_INPUT
    p_tdesc(ctx, gid, 1, 0, &out_desc); // 1 = NOE_TENSOR_TYPE_OUTPUT
    printf("Input:  size=%u bytes, scale=%.2f, zero_point=%d, dtype=%u\n",
           in_desc.size, in_desc.scale, in_desc.zero_point, in_desc.data_type);
    printf("Output: size=%u bytes, scale=%.2f, zero_point=%d, dtype=%u\n",
           out_desc.size, out_desc.scale, out_desc.zero_point, out_desc.data_type);

    int n_dims = out_desc.size / 2;  // INT16 = 2 bytes each
    printf("Embedding dimensions: %d\n\n", n_dims);

    // Create job (MUST pass valid config — NULL segfaults!)
    job_config_npu_t npu_cfg = {0};
    job_config_t cfg = { .conf_j_npu = &npu_cfg };
    uint64_t jid = 0;
    p_create(ctx, gid, &jid, &cfg);

    // Prepare CLIP tokens: [SOT, "hello", "world", EOT, 0, 0, ..., 0]
    // SOT (start of text) = 49406, EOT (end of text) = 49407
    // "hello" = 3306, "world" = 1002
    int32_t tokens[77] = {0};
    tokens[0] = 49406;  // <|startoftext|>
    tokens[1] = 3306;   // "hello"
    tokens[2] = 1002;   // "world"
    tokens[3] = 49407;  // <|endoftext|>

    // Load input and run inference
    p_ltensor(ctx, jid, 0, tokens);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ret = p_infer(ctx, jid, 10000);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("Inference: 0x%x (%s) in %.2f ms\n", ret, ret == 0 ? "success" : "FAILED", ms);

    if (ret == 0) {
        // Read and dequantize output
        void *buf = malloc(out_desc.size);
        p_gtensor(ctx, jid, 1, 0, buf);
        int16_t *raw = (int16_t*)buf;

        // Dequantize: float_val = (int16_val + zero_point) / scale
        printf("\nFirst 10 embedding values (dequantized):\n  ");
        for (int i = 0; i < 10 && i < n_dims; i++) {
            float val = ((float)raw[i] + out_desc.zero_point) / out_desc.scale;
            printf("%.4f ", val);
        }
        printf("...\n");

        // Compute L2 norm
        float norm = 0;
        for (int i = 0; i < n_dims; i++) {
            float val = ((float)raw[i] + out_desc.zero_point) / out_desc.scale;
            norm += val * val;
        }
        printf("L2 norm: %.4f\n", sqrtf(norm));

        free(buf);
    }

    // Cleanup
    p_clean(ctx, jid);
    p_unload(ctx, gid);
    p_deinit(ctx);
    printf("\nDone.\n");
    return 0;
}
```

### Expected Output

```
NPU context initialized
Input:  size=308 bytes, scale=1.00, zero_point=0, dtype=7
Output: size=512 bytes, scale=27.67, zero_point=60, dtype=3
Embedding dimensions: 256

Inference: 0x0 (success) in 17.84 ms

First 10 embedding values (dequantized):
  -0.5456 -0.5459 -0.6198 -1.0267 -0.5273 -0.6199 -0.4997 -0.5364 -0.5363 -0.5459 ...
L2 norm: 15.6243

Done.
```

### Understanding the Numbers

- **Input size 308 bytes** = 77 tokens × 4 bytes/token (INT32)
- **Output size 512 bytes** = 256 values × 2 bytes/value (INT16)
- **Scale 27.67, zero_point 60** — the quantization parameters. Apply: `float = (int16 + 60) / 27.67`
- **17.84ms** — pure NPU hardware inference time (excludes data copy overhead)
- The raw L2 norm is large because these are dequantized quantized values. **Always L2-normalize the output** before using for similarity search.

### CLIP Tokenization

CLIP uses byte-pair encoding (BPE) with a 49,152-token vocabulary. Input must be:
- Padded/truncated to exactly 77 tokens
- Start with token `49406` (`<|startoftext|>`)
- End with token `49407` (`<|endoftext|>`)
- Remaining positions filled with `0` (padding)

The BPE vocabulary file can be downloaded from OpenAI's CLIP repository:
```bash
curl -sL "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz" | gunzip > ~/models/clip_bpe.txt
```

A Python implementation of the tokenizer (~60 lines) is straightforward — see `inference_npu.py` in the model hub or the OpenAI CLIP source.

## CLIP Visual Embeddings

The visual half of CLIP ViT-B/32 — converts images to 512-dimensional vectors in the same embedding space as the text encoder. Enables cross-modal search (find images matching a text description, or vice versa).

### Download

```bash
curl -sL -o ~/models/clip_visual.cix \
  "https://modelscope.cn/models/cix/ai_model_hub_25_Q3/resolve/master/models/Generative_AI/Image_to_Text/onnx_clip/clip_visual.cix"
# ~88MB
```

### Model Specs

| Property | Value |
|----------|-------|
| Architecture | CLIP ViT-B/32 visual encoder |
| Parameters | ~63M |
| Model size | 88MB (INT8 quantized) |
| Input | `[3, 224, 224]` UINT8 — **CHW layout** (channels first) |
| Output | `[512]` UINT8 — image embeddings (quantized) |
| Input size | 150,528 bytes |
| Output size | 512 bytes |
| Input scale | 64.75, zero_point=12 |
| Output scale | 29.52, zero_point=-80 |
| Inference time | ~14.1ms |

### Input Preprocessing

**⚠️ Critical: The model expects CHW (channels-first) layout, not HWC.** This is not documented in the model hub. The wrong layout produces valid-looking but meaningless embeddings.

Standard CLIP preprocessing:

```python
from PIL import Image
import numpy as np

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

def preprocess_clip_image(image_path):
    img = Image.open(image_path).convert('RGB')
    
    # Resize shortest side to 224, center crop to 224x224
    w, h = img.size
    scale = 224 / min(w, h)
    img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    left = (img.width - 224) // 2
    top = (img.height - 224) // 2
    img = img.crop((left, top, left+224, top+224))
    
    # Normalize with CLIP statistics
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    
    # Quantize to U8 and convert to CHW
    arr_u8 = np.clip(arr * 64.75 + 12, 0, 255).astype(np.uint8)
    return arr_u8.transpose(2, 0, 1).flatten()  # CHW!
```

### Output Dequantization

```python
# Raw output: 512 UINT8 values
# Dequantize: float = (uint8_val - 80) / 29.52
embedding = (raw_u8.astype(np.float32) - 80) / 29.52

# L2-normalize for cosine similarity
embedding /= np.linalg.norm(embedding)
```

### Cross-Modal Search

CLIP text and visual encoders share the same embedding space (though different dimensions: 256 for text, 512 for visual). To compare them, project or pad to matching dimensions, or use separate Qdrant collections with shared semantic meaning.

## YOLOv8n Object Detection

Real-time object detection with 80 COCO classes at 66+ fps on the NPU.

### Download

```bash
curl -sL -o ~/models/yolov8n.cix \
  "https://modelscope.cn/models/cix/ai_model_hub_25_Q3/resolve/master/models/ComputeVision/Object_Detection/onnx_yolov8_n/yolov8n.cix"
# ~4.5MB (yes, really)
```

### Model Specs

| Property | Value |
|----------|-------|
| Architecture | YOLOv8 Nano |
| Parameters | ~3.2M |
| Model size | 4.5MB |
| Input | `[3, 640, 640]` INT8 — **BGR CHW layout** |
| Output | `[84, 8400]` FLOAT16 — detections |
| Input size | 1,228,800 bytes |
| Output size | 1,411,200 bytes |
| Input scale | 255.0, zero_point=0 |
| Classes | 80 (COCO) |
| Inference time | ~15.1ms |

### Input Preprocessing

```python
def preprocess_yolo(image_path):
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    
    # Letterbox resize to 640x640 (maintain aspect ratio)
    scale = min(640/orig_w, 640/orig_h)
    new_w, new_h = int(orig_w*scale), int(orig_h*scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    
    padded = Image.new('RGB', (640, 640), (114, 114, 114))
    pad_x = (640 - new_w) // 2
    pad_y = (640 - new_h) // 2
    padded.paste(img, (pad_x, pad_y))
    
    # BGR CHW uint8
    arr = np.array(padded)[:, :, ::-1]  # RGB→BGR
    return arr.transpose(2, 0, 1).flatten()  # HWC→CHW
```

### Output Parsing

The output is 705,600 float16 values reshaped as `(84, 8400)`:
- **84** = 4 bbox coordinates (cx, cy, w, h in pixel space) + 80 class probabilities
- **8400** = detection candidates across all feature map scales

```python
raw = np.frombuffer(output_bytes, dtype=np.float16).astype(np.float32)
pred = raw.reshape(84, 8400).T  # → (8400, 84)

boxes = pred[:, :4]         # cx, cy, w, h
class_probs = pred[:, 4:]   # 80 class scores

# Filter by confidence + apply NMS
max_conf = class_probs.max(axis=1)
mask = max_conf > 0.3
# ... standard NMS follows
```

### Example Output

```
bus.jpg (14.5ms): 4 detections
  person          0.88  [ 219,  405,  344,  857]
  person          0.88  [  50,  395,  243,  911]
  bus             0.84  [  26,  229,  803,  743]
  person          0.80  [ 667,  380,  810,  876]
```

## Performance Numbers

All benchmarks run on Orange Pi 6 Plus (CIX CD8180, Zhouyi Z3, 3 cores × 4 TECs). Numbers from 200 iterations after 3-iteration warmup.

A benchmark tool is included at [`tools/bench_npu.c`](tools/bench_npu.c):

```bash
gcc -O2 -o bench_npu tools/bench_npu.c -ldl -lm
LD_LIBRARY_PATH=/usr/share/cix/lib ./bench_npu 200 /path/to/clip_txt.cix
```

### Raw NPU Inference (200 iterations each)

| Model | Mean | P50 | P95 | Min | Max | Stddev | Throughput |
|-------|------|-----|-----|-----|-----|--------|------------|
| CLIP Text (63M, 71MB) | 15.52 ms | 15.76 ms | 16.57 ms | 14.01 ms | 16.80 ms | 0.74 ms | **64.4 inf/s** |
| CLIP Visual (63M, 88MB) | 14.11 ms | 14.13 ms | 15.38 ms | 13.22 ms | 15.89 ms | 0.57 ms | **70.9 inf/s** |
| YOLOv8n (3.2M, 4.5MB) | 15.14 ms | 15.38 ms | 15.83 ms | 13.85 ms | 16.37 ms | 0.63 ms | **66.1 inf/s** |

### Full Pipeline (create job + load tensor + infer + cleanup)

| Model | Mean | P50 | P95 | Throughput |
|-------|------|-----|-----|------------|
| CLIP Text | 17.47 ms | 17.55 ms | 18.12 ms | **57.2 inf/s** |
| CLIP Visual | 21.76 ms | 21.55 ms | 25.51 ms | **46.0 inf/s** |
| YOLOv8n | 21.22 ms | 20.80 ms | 24.42 ms | **47.1 inf/s** |

The visual/YOLO models have higher full-pipeline overhead due to larger input tensors (150KB and 1.2MB vs 308 bytes for text).

### End-to-End HTTP Comparison (NPU vs CPU)

Apples-to-apples comparison including HTTP server overhead, tokenization, and JSON serialization. 100 iterations each.

| | NPU (CLIP ViT-B/32) | CPU (all-MiniLM-L6-v2) |
|---|---|---|
| **Architecture** | 12-layer transformer | 6-layer transformer |
| **Parameters** | ~63M | ~22M |
| **Model size** | 71 MB (INT8/INT16) | 24 MB (Q8_0 GGUF) |
| **Output dims** | 256 | 384 |
| **Mean latency** | 19.73 ms | 14.52 ms |
| **P50 latency** | 19.76 ms | 14.27 ms |
| **P95 latency** | 21.39 ms | 16.39 ms |
| **Throughput** | 50.7 infer/sec | 68.9 infer/sec |
| **CPU usage** | **~0%** (NPU silicon) | 4 cores @ 100% |
| **Server RAM** | ~113 MB (Python wrapper) | ~60 MB (llama-server) |
| **Runtime** | Python + ctypes + libnoe | llama-server (C++) |

### Analysis

For this specific small model comparison, **the CPU is faster** — MiniLM is 3× smaller and llama-server is highly optimized C++ with ARM NEON/SVE. However:

1. **The NPU runs a 3× larger model at comparable speed.** CLIP ViT-B/32 has 63M params vs MiniLM's 22M — normalizing for model size, the NPU is significantly more efficient.

2. **Zero CPU impact.** During NPU inference, all 12 CPU cores are completely free. This matters when the CPU is handling other workloads (web servers, databases, system tasks).

3. **Consistent latency.** NPU stddev is 0.74ms vs CPU's wider variance (13.38–19.76ms range). The NPU has no contention with other processes.

4. **Scales with model size.** The NPU's advantage grows with larger models where CPU inference becomes impractical. The same NPU handles vision models (ResNet, YOLO) at similar speeds.

### Estimated NPU Performance

Based on CLIP ViT-B/32 text encoder (~1.2 GFLOPS per inference):

| Metric | Value |
|--------|-------|
| Sustained throughput | ~77 GFLOPS (INT8) |
| Embedding dims/sec | ~16,500 dims/sec |
| Inferences/sec | 64.4 (raw) / 50.7 (HTTP) |

## Building an Embedding Server

For production use, wrap the NPU inference in an HTTP server. A Python implementation using `ctypes` to call `libnoe.so` is the simplest approach:

1. Load `libnoe.so` via `ctypes.CDLL()`
2. Initialize NPU context and load model at startup
3. For each request: tokenize text → create job → load tensor → infer → dequantize → return JSON

Key considerations:
- **Job lifecycle:** Create a new job per request and clean it after. Jobs hold NPU memory and can't be reused after inference.
- **Concurrency:** `libnoe.so` is NOT thread-safe. Use a single-threaded server or serialize NPU access with a lock.
- **Warmup:** First inference after model load takes ~50ms (NPU initialization). Subsequent inferences are ~18ms.
- **Memory:** The model stays loaded in NPU memory between inferences. Only job buffers are allocated/freed per request.

### Systemd Service

```ini
[Unit]
Description=NPU Embedding Server
After=network.target

[Service]
Type=simple
User=your-user
Environment=PORT=8091
Environment=MODEL_PATH=/path/to/clip_txt.cix
Environment=BPE_PATH=/path/to/clip_bpe.txt
ExecStart=/usr/bin/python3 /path/to/server.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

## Quantization & Dequantization

Models compiled for the NPU use INT8 or INT16 quantization. The `tensor_desc_t` struct provides the parameters:

### Input Quantization

```
quantized_value = round(float_value × scale - zero_point)
quantized_value = clamp(quantized_value, dtype_min, dtype_max)
```

For CLIP text input: `scale=1.0, zero_point=0, dtype=INT32` — just pass raw token IDs directly (no quantization needed).

### Output Dequantization

```
float_value = (quantized_value + zero_point) / scale
```

For CLIP text output: `scale=27.67, zero_point=60, dtype=INT16`:
```c
int16_t *raw = (int16_t*)output_buffer;
for (int i = 0; i < 256; i++) {
    float val = ((float)raw[i] + 60) / 27.6738f;
    embedding[i] = val;
}
```

**Always L2-normalize** the dequantized embeddings before cosine similarity:
```c
float norm = 0;
for (int i = 0; i < dims; i++) norm += embedding[i] * embedding[i];
norm = sqrtf(norm);
for (int i = 0; i < dims; i++) embedding[i] /= norm;
```

## Alternative Paths

### ONNX Runtime with Zhouyi Execution Provider

CIX ships an ONNX Runtime build with `ZhouyiExecutionProvider` at `/usr/share/cix/pypi/`. This compiles ONNX models to NPU graphs at runtime — no manual `.cix` compilation needed. However:

- **Requires Python 3.11-3.12** (wheel incompatible with 3.13+)
- **Static shapes only** — model must be exported with fixed dimensions
- **Limited op support** — Shape, Gather, Slice, Where, Concat, Cast are unsupported
- Unsupported ops fall back to CPU automatically
- BERT/transformer models trigger fallback on most ops, making this path inefficient for NLP

For vision models (ResNet, MobileNet, YOLO), this path works well. For NLP, use pre-compiled `.cix` models from the model hub.

### CixBuilder (Custom Model Compilation)

CixBuilder (`cixbuild`) is the official ONNX → `.cix` compiler, distributed as a Python wheel in the CIX NOE SDK (`CixBuilder-6.1.3407.2-cp310-none-linux_x86_64.whl`). It wraps the Compass toolchain (Arm China's AIPUBuilder).

**Requirements:**
- **x86_64 Linux only** (not ARM)
- **Python 3.10** (exact — the wheel is cp310-specific)
- Heavy dependencies: PyTorch, TensorFlow, ONNX, NumPy
- `LD_LIBRARY_PATH` must include `AIPUBuilder/simulator-lib/` for the GBuilder phase
- Conda/Miniforge recommended for isolation

**Pipeline:** ONNX → Parse → Optimize/Quantize → GBuilder → `.cix`

**Config file format** (`.cfg`):
```ini
[Common]
mode = build

[Parser]
input_model = /path/to/model.onnx      # Key is "input_model", NOT "model_path"
model_type = onnx
model_name = my-model                   # Required — KeyError if missing
output = output_tensor_name
output_dir = /path/to/output

[Optimizer]
dataset = numpymultiinputdataset        # Must be a built-in dataset name
data = /path/to/calibration/dir
calibration_data = /path/to/calib.npz   # .npz for multi-input models
calibration_batch_size = 1
output_dir = /path/to/output

[GBuilder]
target = Z3_1204                        # Z3 with 12 TECs (3 cores × 4)
outputs = /path/to/output/model.cix     # Key is "outputs" (plural)
```

**Supported Z3 targets:** `Z3_0901`, `Z3_1002`, `Z3_1104`, `Z3_1204`

**Known limitations (v6.1.3407):**
- **Transformer/attention models fail on Z3**: All MatMul operations in attention layers (Q/K/V projections, FFN) are mapped to "Convolution" ops that the Z3 GBuilder backend doesn't support. Every attention and FFN layer produces `Unsupport node type: Convolution` errors.
- Vision models (CNNs: ResNet, YOLO, CLIP visual) compile fine — these use actual convolutions.
- CLIP text encoder works because CIX likely used an internal/newer toolchain version.
- The pre-compiled `.cix` models in the CIX AI Model Hub were built with tooling not publicly available.
- `output_dir` is NOT a valid GBuilder parameter (causes "Unknown option" error)
- Parser reorders multi-input tensors alphabetically — check IR output order

**Calibration data for multi-input models:**
- Use `.npz` format with keys `input0`, `input1`, `input2`, etc.
- Shape: `[N, ...original_shape]` where N = number of calibration samples
- For text models: random token IDs as int32 (int64 auto-converted to int32)

**Bottom line:** As of SDK v25 Q3, you cannot compile custom transformer/NLP models to `.cix` for the Z3 NPU. Use pre-compiled models from the CIX Model Hub, or run NLP models on CPU.

### NOE Python Bindings

A Python wrapper (`libnoe` wheel) exists but requires Python `<3.13,>=3.11`. If you have a compatible Python version:

```python
from libnoe import NPU, noe_create_job_cfg_t, NOE_TENSOR_TYPE_INPUT, NOE_TENSOR_TYPE_OUTPUT

npu = NPU()
npu.noe_init_context()
_, graph_id = npu.noe_load_graph("model.cix")
cfg = noe_create_job_cfg_t()
_, job_id = npu.noe_create_job(graph_id, cfg)
npu.noe_load_tensor(job_id, 0, input_bytes)
npu.noe_job_infer_sync(job_id, -1)
_, output = npu.noe_get_tensor(job_id, NOE_TENSOR_TYPE_OUTPUT, 0)
```

## Troubleshooting

### Segfault in `noe_create_job`
You're passing NULL as the config parameter. Pass a zero-initialized `job_config_t` struct. See [Critical ABI Gotcha](#critical-abi-gotcha).

### `noe_load_graph` returns 0xC (OPEN_FILE_FAIL)
Model file doesn't exist or isn't readable. Check the path and permissions.

### `noe_load_graph` returns 0x9 (INVALID_GBIN)
The `.cix` file is corrupted or is a git LFS pointer (133 bytes) instead of the actual model. Re-download using the direct URL method.

### `noe_job_infer_sync` returns 0x11 (JOB_EXCEPTION)
Input data doesn't match expected size. Check `tensor_desc_t.size` and ensure your input buffer is exactly that many bytes.

### Inference returns all zeros
Input tokens aren't CLIP-formatted. Ensure:
- Token 0 is `49406` (start of text)
- Text tokens follow
- Last meaningful token is `49407` (end of text)
- Remaining tokens are `0` (padding)
- Total length is exactly 77

### "pure virtual method called" on exit
Happens when the Python process exits without calling `noe_deinit_context()`. Not harmful — the kernel driver cleans up NPU resources. Fix by adding proper cleanup in a `finally` block or signal handler.

---

## License

The test code and documentation in this file are released under Apache-2.0.

The CIX model hub models have their own licenses (typically MIT or Apache-2.0) — check each model's LICENSE file.

The `libnoe.so` library and NPU firmware are proprietary CIX/ARM China binaries distributed in their Debian packages.
