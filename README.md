# NPU Driver Guide — Orange Pi 6 Plus (CIX CD8180 / Zhouyi V3)

*Last updated: 2026-02-10*
*Status: Driver loaded, 3 models running on NPU — CLIP text/visual + YOLOv8n object detection*
*Tested on: Armbian 26.2.0-trunk.410, kernel 6.18.9-current-arm64*

## Attribution

This driver is based on the **ARM China Zhouyi AIPU kernel driver** from:

- **Original source:** [orangepi-xunlong/component_cix-current](https://github.com/orangepi-xunlong/component_cix-current) (`cix_opensource/npu/npu_driver/driver/`)
- **Original authors:** Copyright (c) 2023-2024 Arm Technology (China) Co. Ltd.
- **License:** GPL-2.0 (see [LICENSE](LICENSE))

This repository contains the original driver source with **6 patches** applied for compatibility with mainline Armbian kernel 6.18.9 (arm64). All modifications are documented below. No proprietary code is included.

---

## Real-World Example: Edge Anomaly Detection

To give a concrete sense of what the NPU is actually doing:

A background agent collects system health metrics every 30 seconds — CPU usage, RAM, load average, temperatures, disk — and turns them into a text description like:

> *"host mybox cpu 2.4% ram 25.5% load 0.46 temps [soc 44°C] disks [/ 19.7%]"*

That text gets sent to the NPU, which runs it through a CLIP text encoder (63M parameter transformer, 12 layers) to produce a **256-dimensional vector** — a numerical "fingerprint" of the system's current state.

The vector is shipped to a remote server running a vector database (Qdrant). An anomaly detector compares each new vector against the last 20 vectors using cosine similarity. If the system state suddenly looks different from recent history — say CPU spikes from 2% to 95% — the embedding diverges from the baseline, triggering an alert.

**The NPU's role: convert system health snapshots into vectors for real-time anomaly detection.** It's the "is this normal?" engine.

### Why use the NPU instead of CPU?

| | NPU | CPU |
|---|---|---|
| Model | CLIP ViT-B/32 (63M params) | all-MiniLM-L6-v2 (22M params) |
| Latency | 15.5ms | 14.5ms |
| Throughput | 64 infer/sec | 69 infer/sec |
| CPU cores used | **0** | 4 cores @ 100% |

The NPU runs a **3× larger model** at the same speed while leaving all 12 CPU cores completely free. This matters when the device is also running web servers, databases, or other workloads. The NPU is dedicated silicon — it doesn't compete for resources.

---

## NPU Benchmark Suite

Three pre-compiled models from the [CIX AI Model Hub](https://modelscope.cn/models/cix/ai_model_hub_25_Q3), all running on the Zhouyi Z3 NPU with **zero CPU load**:

### Models Tested

| Model | Task | Params | Size | Input | Output |
|-------|------|--------|------|-------|--------|
| CLIP ViT-B/32 Text | Text → embedding | 63M | 71 MB | 77 INT32 tokens (308 B) | 256-dim INT16 (512 B) |
| CLIP ViT-B/32 Visual | Image → embedding | 63M | 88 MB | 224×224×3 U8 CHW (150 KB) | 512-dim U8 (512 B) |
| YOLOv8n | Object detection | 3.2M | 4.5 MB | 640×640×3 U8 CHW (1.2 MB) | 84×8400 F16 (1.4 MB) |

### Raw NPU Inference (200 iterations each)

| Model | Mean | P50 | P95 | Min | Max | Throughput |
|-------|------|-----|-----|-----|-----|------------|
| **CLIP Text** | 15.52 ms | 15.76 ms | 16.57 ms | 12.52 ms | 16.92 ms | **64.4 inf/s** |
| **CLIP Visual** | 14.11 ms | 14.13 ms | 15.38 ms | 13.22 ms | 15.89 ms | **70.9 inf/s** |
| **YOLOv8n** | 15.14 ms | 15.38 ms | 15.83 ms | 13.85 ms | 16.37 ms | **66.1 inf/s** |

### Full Pipeline (create job + load tensor + infer + cleanup)

| Model | Mean | P50 | P95 | Throughput |
|-------|------|-----|-----|------------|
| **CLIP Text** | 17.47 ms | 17.55 ms | 18.12 ms | **57.2 inf/s** |
| **CLIP Visual** | 21.76 ms | 21.55 ms | 25.51 ms | **46.0 inf/s** |
| **YOLOv8n** | 21.22 ms | 20.80 ms | 24.42 ms | **47.1 inf/s** |

### NPU vs CPU: Why Dedicated AI Silicon Matters

| Metric | NPU (Zhouyi Z3) | CPU (Cortex-A720) |
|--------|-----------------|-------------------|
| CLIP Text Embedding | 15.5 ms / 64 inf/s | 14.5 ms / 69 inf/s ¹ |
| CLIP Visual Embedding | 14.1 ms / 71 inf/s | N/A ² |
| YOLOv8n Detection | 15.1 ms / 66 inf/s | N/A ² |
| Model (text embed) | CLIP ViT-B/32 (**63M** params) | all-MiniLM-L6-v2 (**22M** params) ¹ |
| CPU cores consumed | **0** | 4–8 cores @ 100% |
| RAM overhead | ~113 MB (shared driver) | ~60 MB per model |
| Concurrent models | All 3 models, same NPU | Each model competes for cores |
| Power | Dedicated low-power silicon | Full CPU power domain |

> ¹ CPU comparison uses all-MiniLM-L6-v2 (22M params, 384-dim) via llama-server — a **3× smaller model** that produces less-expressive embeddings. Running the full CLIP ViT-B/32 (63M params) on CPU would be significantly slower.
>
> ² No CPU benchmarks for CLIP Visual or YOLOv8n — these models require ONNX Runtime or PyTorch on CPU, which would consume all available cores and take 50-200ms+ per inference on ARM.

**The key insight:** The NPU doesn't just offload work — it enables running models that would be impractical on CPU. Three AI models running simultaneously with zero CPU impact, at 60+ inferences per second each.

### Estimated NPU Compute

Based on CLIP ViT-B/32 (63M params, ~126M INT8 MACs per inference at 15.5ms):
- **~8 GOPS INT8** sustained throughput (conservative estimate)
- Theoretical peak for 3 cores × 4 TECs: significantly higher
- Real-world efficiency limited by memory bandwidth and model structure

### Practical Example: YOLOv8n Object Detection

Running on the classic YOLO test image (`bus.jpg`):

```
bus.jpg (14.5ms): 4 detections
  person          0.88  [ 219,  405,  344,  857]
  person          0.88  [  50,  395,  243,  911]
  bus             0.84  [  26,  229,  803,  743]
  person          0.80  [ 667,  380,  810,  876]
```

3 people and 1 bus detected with 80-88% confidence at 69 fps — fast enough for real-time video processing on an edge device.

### Practical Example: CLIP Image Similarity

CLIP visual embeddings capture semantic image content. Cosine similarity between different test images:

```
              orange      blue     green   stripes     noise       bus
orange        1.000     0.825    0.848    0.828     0.771     0.645
blue          0.825     1.000    0.826    0.762     0.770     0.626
green         0.848     0.826    1.000    0.833     0.767     0.621
stripes       0.828     0.762    0.833    1.000     0.745     0.604
noise         0.771     0.770    0.767    0.745     1.000     0.624
bus (photo)   0.645     0.626    0.621    0.604     0.624     1.000
```

The real photograph clearly separates from synthetic test images (0.60-0.65 similarity vs 0.75-0.85 within synthetics). Solid colors cluster together. These embeddings can be used for image search, visual anomaly detection, and cross-modal retrieval (find images matching a text description).

For the inference API, model loading, and code examples, see [INFERENCE.md](INFERENCE.md).

---

## Hardware

- **Board:** Orange Pi 6 Plus
- **SoC:** CIX CD8180 (aka "CIX Phecda Board" in DMI)
- **NPU:** ARM China Zhouyi Z3 (codename "Compass"), platform "SKY1"
- **NPU Config:** 1 partition, 1 cluster, 3 cores, 4 TECs per core (12 tensor execution channels)
- **ACPI Device:** `CIXH4000:00` (IOMMU group 9)
- **Kernel tested:** Armbian 6.18.9-current-arm64 (trunk.410)

## Source Repos

All source code comes from Orange Pi's GitHub (`orangepi-xunlong`):

| Repo | Purpose |
|------|---------|
| `component_cix-current` | NPU kernel driver (open source), userspace debs (proprietary), firmware |
| `orangepi-build` | Full build system, board configs, family configs |

**Key paths in `component_cix-current`:**
- `cix_opensource/npu/npu_driver/driver/` — kernel module source (GPL)
- `debs/` — prebuilt arm64 .deb packages (llama-cpp, MNN, NOE runtime, etc.)
- `cix_proprietary/` — proprietary blobs and firmware

## Prerequisites

```bash
# Kernel headers (match your running kernel!)
sudo apt install linux-headers-current-arm64

# Build tools
sudo apt install build-essential
```

## Step 1: Clone the Driver Source

```bash
mkdir -p ~/projects/npu-driver
cd ~/projects/npu-driver
git clone --depth 1 --sparse https://github.com/orangepi-xunlong/component_cix-current.git .
git sparse-checkout set cix_opensource/npu debs
```

This gives you the kernel driver source and prebuilt userspace packages (~10MB total).

## Step 2: Rewrite the Makefile

The original Makefile relies on environment variables (`BUILD_TARGET_PLATFORM_KMD`, `BUILD_AIPU_VERSION_KMD`, etc.) that **get lost during kbuild's recursive make invocation**. This causes:
- `sky1.c` to be compiled as a separate module instead of part of `aipu.ko`
- The `init_module` symbol to be missing from the final binary
- The module loads silently and does absolutely nothing

**⚠️ GOTCHA #1: The original Makefile does NOT work for out-of-tree builds.**

Replace `cix_opensource/npu/npu_driver/driver/Makefile` entirely:

```makefile
# SPDX-License-Identifier: GPL-2.0
# Modified for Orange Pi 6 Plus (CIX CD8180 / SKY1 / Zhouyi V3) on Armbian

ifneq ($(KERNELRELEASE),)

ccflags-y += -DCONFIG_SKY1 \
             -DCONFIG_ARMCHINA_NPU_ARCH_V3 \
             -DBUILD_ZHOUYI_V3 \
             -DKMD_VERSION=\"5.11.0\" \
             -I$(src)/armchina-npu/ \
             -I$(src)/armchina-npu/include \
             -I$(src)/armchina-npu/zhouyi

obj-m := aipu.o
aipu-y := armchina-npu/sky1/sky1.o \
          armchina-npu/aipu.o \
          armchina-npu/aipu_common.o \
          armchina-npu/aipu_io.o \
          armchina-npu/aipu_irq.o \
          armchina-npu/aipu_job_manager.o \
          armchina-npu/aipu_mm.o \
          armchina-npu/aipu_dma_buf.o \
          armchina-npu/aipu_priv.o \
          armchina-npu/aipu_tcb.o \
          armchina-npu/zhouyi/zhouyi.o \
          armchina-npu/zhouyi/v3.o \
          armchina-npu/zhouyi/v3_priv.o

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean

endif
```

**Why this works:** Uses `ccflags-y` (kbuild-native) instead of `EXTRA_CFLAGS`, uses `$(src)` for include paths (resolved correctly during recursion), uses `aipu-y` (the kbuild composite module syntax), and hardcodes the platform/architecture instead of relying on env vars.

## Step 3: Patch Source Files for Kernel 6.x Compatibility

The driver was written for Orange Pi's custom 6.1 kernel. Six patches are needed for mainline 6.18+:

### Patch 1: Wrap devfreq functions (sky1.c)

The devfreq code calls CIX-specific SCMI functions (`scmi_device_get_freq`, `scmi_device_set_freq`, `scmi_device_opp_table_parse`) that don't exist in mainline kernels. The *call sites* are already guarded by `#ifdef CONFIG_ENABLE_DEVFREQ`, but the *function definitions* are not — and GCC 14+ treats implicit function declarations as hard errors.

**⚠️ GOTCHA #2: GCC 14+ makes implicit function declarations a hard error, not a warning.**

In `armchina-npu/sky1/sky1.c`, wrap the devfreq function definitions:

```c
// Before the line: static int sky1_npu_devfreq_target(...)
#ifdef CONFIG_ENABLE_DEVFREQ

// ... all devfreq functions ...

// After the closing brace of sky1_npu_devfreq_remove():
#endif /* CONFIG_ENABLE_DEVFREQ */
```

This wraps roughly lines 67-230 (the block from `sky1_npu_devfreq_target` through `sky1_npu_devfreq_remove`).

### Patch 2: Fix platform_driver.remove return type (sky1.c)

Kernel 6.2+ changed `platform_driver.remove` callback from `int` to `void`.

In `armchina-npu/sky1/sky1.c`:
```c
// Change:
static int sky1_npu_remove(struct platform_device *p_dev)
// To:
static void sky1_npu_remove(struct platform_device *p_dev)

// And change the return at the end:
// return 0;
// To:
// return;
```

### Patch 3: Fix MODULE_IMPORT_NS syntax (aipu_dma_buf.c)

Kernel 6.4+ changed `MODULE_IMPORT_NS` to take a string argument.

In `armchina-npu/aipu_dma_buf.c`:
```c
// Change:
MODULE_IMPORT_NS(DMA_BUF);
// To:
MODULE_IMPORT_NS("DMA_BUF");
```

### Patch 4: Bypass IOVA reservation (aipu_mm.c)

The `aipu_mm_reserved_iova_for_never_map` function accesses `iommu_domain->iova_cookie` which is a private/opaque struct (`struct iommu_dma_cookie`). Its internal layout changed across kernel versions, causing a spinlock crash on kernels 6.4+.

**⚠️ GOTCHA #3: First insmod attempt will crash the kernel module if this isn't patched. The crashed module gets stuck in `initstate=coming` and can't be removed — requires a reboot.**

In `armchina-npu/aipu_mm.c`, make the function a no-op for newer kernels:

```c
static int aipu_mm_reserved_iova_for_never_map(struct aipu_memory_manager *mm, bool flag)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0)
    /* iommu_dma_cookie internals not accessible on newer kernels; skip IOVA reservation */
    return 0;
}
#else
    // ... original function body ...
    return 0;
}
#endif
```

The NPU works fine without this reservation — it just means slightly less optimal IOVA memory layout.

### Patch 5: Raw modinfo entries (sky1.c)

**⚠️ GOTCHA #4: In kernel 6.18, multi-file out-of-tree modules have a modpost bug where `MODULE_LICENSE`, `MODULE_DESCRIPTION`, and `MODULE_IMPORT_NS` macros prepend `KBUILD_MODNAME.` to .modinfo entries, but modpost searches for bare `tag=value`. This causes "missing MODULE_LICENSE" and "does not import namespace DMA_BUF" errors.**

The workaround is to add raw `.modinfo` section entries that bypass the `MODULE_INFO` macro. In `armchina-npu/sky1/sky1.c`, comment out the original macros and add:

```c
/* Comment out or remove these:
MODULE_LICENSE("GPL v2");
MODULE_IMPORT_NS("DMA_BUF");
*/

/* Raw modinfo entries for modpost compatibility */
static const char __modinfo_license[]
    __used __section(".modinfo") __aligned(1)
    = "license=GPL";
static const char __modinfo_description[]
    __used __section(".modinfo") __aligned(1)
    = "description=ARM China AIPU NPU driver for CIX SKY1";
static const char __modinfo_import_ns[]
    __used __section(".modinfo") __aligned(1)
    = "import_ns=DMA_BUF";
```

### Patch 6: Replace module_platform_driver with explicit init/exit (sky1.c)

For debugging (and to get proper `pr_info` output), replace:

```c
// Remove:
module_platform_driver(aipu_platform_driver);

// Add:
static int __init aipu_init(void)
{
    int ret;
    pr_info("aipu: registering platform driver\n");
    ret = platform_driver_register(&aipu_platform_driver);
    pr_info("aipu: platform_driver_register returned %d\n", ret);
    return ret;
}

static void __exit aipu_exit(void)
{
    pr_info("aipu: unregistering platform driver\n");
    platform_driver_unregister(&aipu_platform_driver);
}

module_init(aipu_init);
module_exit(aipu_exit);
```

This is technically optional (the Makefile rewrite in Step 2 fixes the root cause of `init_module` being missing), but it's good practice for debugging.

## Step 4: Build

```bash
cd ~/projects/npu-driver/cix_opensource/npu/npu_driver/driver
make -C /lib/modules/$(uname -r)/build M=$(pwd) clean
make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
```

Expected output ends with:
```
LD [M]  aipu.ko
BTF [M] aipu.ko
Skipping BTF generation for aipu.ko due to unavailability of vmlinux
```

**Verify `init_module` exists:**
```bash
nm aipu.ko | grep init_module
# Should show: 0000000000000008 T init_module
```

If `init_module` is missing, the Makefile rewrite wasn't applied correctly. The module will load silently and do nothing.

## Step 5: Install

```bash
sudo mkdir -p /lib/modules/$(uname -r)/extra
sudo cp aipu.ko /lib/modules/$(uname -r)/extra/
sudo depmod -a
```

## Step 6: Load

```bash
sudo modprobe aipu
```

Check dmesg for successful probe:
```
aipu: registering platform driver
armchina CIXH4000:00: sky1_npu_probe: NPU core num is 3
armchina CIXH4000:00: AIPU KMD (v5.11.0) probe start...
armchina CIXH4000:00: AIPU detected: zhouyi-v3
armchina CIXH4000:00: ############# ZHOUYI V3 AIPU #############
armchina CIXH4000:00: # Core Count per Cluster: 3
armchina CIXH4000:00: # TEC Count per Core: 4
armchina CIXH4000:00: ##########################################
```

Verify:
```bash
ls -la /dev/aipu          # Should exist: crw-rw-rw- root root
ls /sys/bus/platform/drivers/armchina/  # Should show CIXH4000:00
```

## Step 7: Install Userspace Packages

```bash
cd ~/projects/npu-driver

# NOE Runtime (NPU userspace driver) — will error on Python 3.13+ but .so files install fine
sudo dpkg --force-all -i debs/cix-noe-umd_2.0.2_arm64.deb

# llama.cpp (CPU-only build, NPU backend not included)
sudo dpkg -i debs/cix-llama-cpp_1.0.0+2503.radxa_arm64.deb

# MNN inference framework
sudo dpkg -i debs/cix-mnn_1.0.0+2503.radxa_arm64.deb

# Add CIX libraries to linker path
sudo bash -c 'echo "/usr/share/cix/lib" > /etc/ld.so.conf.d/cix.conf'
sudo ldconfig
```

**⚠️ GOTCHA #5: `cix-noe-umd` postinst script fails because it tries to pip-install a Python wheel that requires Python `<3.13,>=3.11`. Debian forky ships Python 3.13.11 which is outside this range. The C shared libraries (`libnoe.so`, `libnoe.so.0`, `libnoe.so.0.6.0`, `libnoe.a`) install correctly despite the error. Use `--force-all` to proceed.**

**⚠️ GOTCHA #6: The prebuilt `cix-llama-cpp` does NOT use the NPU at all — `--list-devices` shows nothing. It's a standard CPU-only build (ARM NEON/SVE). There is no public NPU backend for llama.cpp on Zhouyi Z3. Hypothetically you'd need to write a custom GGML backend linking against `libnoe.so`, but this doesn't exist. The MNN framework is another option CIX mentions but no NPU-accelerated LLM path is publicly available.**

## Rebuilding After Kernel Updates

The `aipu.ko` module is built against a specific kernel version. After any kernel update (e.g., `apt upgrade` pulling a new `linux-image-current-arm64`), you must rebuild and reinstall:

```bash
cd ~/projects/npu-driver/cix_opensource/npu/npu_driver/driver
make clean
make -j6
sudo cp aipu.ko /lib/modules/$(uname -r)/extra/
sudo depmod -a
sudo modprobe aipu
```

This takes under a minute on the Orange Pi 6 Plus. Verify with `ls /dev/aipu` and `dmesg | grep aipu`.

## Verification

### Quick test that NOE runtime connects to NPU:
```c
// test_noe.c
#include <stdio.h>
#include <dlfcn.h>
typedef int (*noe_init_fn)(void**);
int main() {
    void *lib = dlopen("libnoe.so", RTLD_NOW);
    if (!lib) { printf("Failed: %s\n", dlerror()); return 1; }
    noe_init_fn init = dlsym(lib, "noe_init_context");
    void *ctx = NULL;
    int ret = init(&ctx);
    printf("noe_init_context returned: %d (0=success), ctx=%p\n", ret, ctx);
    dlclose(lib);
    return ret;
}
```
```bash
gcc -o test_noe test_noe.c -ldl
LD_LIBRARY_PATH=/usr/share/cix/lib ./test_noe
# Expected: noe_init_context returned: 0 (0=success), ctx=0x...
```

## Files Modified (relative to repo root)

| File | Patches Applied |
|------|----------------|
| `cix_opensource/npu/npu_driver/driver/Makefile` | Complete rewrite (Step 2) |
| `cix_opensource/npu/npu_driver/driver/armchina-npu/sky1/sky1.c` | Patches 1, 2, 5, 6 |
| `cix_opensource/npu/npu_driver/driver/armchina-npu/aipu_dma_buf.c` | Patch 3 |
| `cix_opensource/npu/npu_driver/driver/armchina-npu/aipu_mm.c` | Patch 4 |

## Known Limitations

- **No devfreq:** NPU runs without dynamic frequency scaling (no power management). Acceptable tradeoff.
- **No BTF:** Skipped due to missing vmlinux. Doesn't affect functionality.
- **Python NOE wheel:** Requires Python 3.11-3.12, won't install on 3.13+. C library works fine.
- **llama.cpp:** Prebuilt binary is CPU-only. NPU backend needs separate build.
- **Kernel version locked:** Module must be rebuilt when kernel updates (standard for out-of-tree modules).

## Running Models

Once the driver is loaded and `/dev/aipu` exists, see **[INFERENCE.md](INFERENCE.md)** for:

- Downloading pre-compiled models from CIX's AI model hub
- Running CLIP text embeddings (~18ms/inference, 256 dimensions)
- The NOE C API workflow with full code examples
- Quantization/dequantization details
- Performance benchmarks
- Building an HTTP embedding server
- Critical ABI gotchas (with fixes)

## TODO

- [ ] Set up DKMS for automatic rebuilds on kernel updates
- [x] ~~Test additional models from CIX model hub~~ — CLIP Visual + YOLOv8n confirmed working
- [ ] Benchmark multi-model concurrent inference across NPU cores
- [ ] Investigate NOE Python wheel compatibility fix for Python 3.13
- [ ] Test larger YOLO variants (YOLOv8l, YOLOv12) for accuracy comparison
- [ ] Build cross-modal search demo (text query → find matching images via CLIP)

## Troubleshooting

### Module loads but no dmesg output / no /dev/aipu
Check `nm aipu.ko | grep init_module`. If missing, the Makefile wasn't rewritten correctly. The kbuild system is compiling sky1.c as a separate module.

### insmod crashes with kernel oops in reserve_iova
Patch 4 (IOVA bypass) wasn't applied. The crashed module will be stuck — reboot required.

### modpost errors about DMA_BUF namespace
Apply Patch 5 (raw modinfo entries). The `MODULE_IMPORT_NS` macro doesn't work correctly for multi-file out-of-tree modules in kernel 6.18.

### "missing MODULE_LICENSE" error during build
Same root cause as DMA_BUF — apply Patch 5.

### Module loads, probe runs, but no /dev/aipu
Check if the ACPI device exists: `ls /sys/bus/acpi/devices/CIXH4000:00/`. If not, the firmware/ACPI tables don't expose the NPU (shouldn't happen on Orange Pi 6 Plus).

## See Also

- [CIXBUILDER.md](CIXBUILDER.md) — Compiling custom models for the Z3 NPU (what works, what doesn't)
- [orange-pi-6-plus-gpu](https://github.com/visorcraft/orange-pi-6-plus-gpu) — Mali-G720 (Panthor) GPU bring-up on the same board
