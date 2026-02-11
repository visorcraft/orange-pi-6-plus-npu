# CixBuilder — Compiling Models for the Z3 NPU

*Findings from attempting to compile transformer models on CixBuilder v6.1.3407.*

## Setup

- **Host:** Any x86_64 Linux machine (CixBuilder is x86-only, cross-compiles for ARM)
- **CixBuilder version:** 6.1.3407
- **Target:** `X2_1204` (NOT `Z3` or `Z3_1204` — these don't work)
- **TPC kernels:** 538 kernels extracted from vendor Debian image (`ZhouyiOperators_x2` wheel in `Orangepi6plus_1.0.2_debian_bookworm`)
  - Radxa SDK only ships 2 kernels (constant, repeat) — vendor wheel has all 538
- **NPU hardware:** ARM China Zhouyi Z3, 3 cores × 4 TECs (12 tensor execution channels)

## Target Selection

Working `.cix` models (e.g., `clip_txt.cix`) use target `X2_1204MP3`. The target **must** be `X2_1204`:

- `Z3_1204` — doesn't support Convolution (which MatMul maps to). Fails at GBuilder.
- `Z3` (bare) — invalid target, crashes.
- `X2_1204` — works. CixBuilder cross-compiles X2 targets that run on Z3 hardware.

## What Works

### CNN/Detection Models ✅
- **YOLOv8n** — 4.5MB `.cix`, 15.1ms inference, 66 inf/s, 80-class detection
- These models are well-suited to INT8 quantization

### CLIP Encoders ✅
- **CLIP ViT-B/32 text** — 71MB `.cix`, 15.5ms, 64 inf/s, 256-dim embeddings
- **CLIP ViT-B/32 visual** — 88MB `.cix`, 14.1ms, 71 inf/s, 512-dim embeddings
- Pre-compiled from CIX AI Model Hub (compiled with internal tooling)

## What Doesn't Work

### BERT/Transformer Embedding Models ❌

Tested: **all-MiniLM-L6-v2** (sentence-transformers)

#### Compilation: Success (with workarounds)
- CixBuilder compiles it: 0 errors, 34MB `.cix`, runs at 221 inf/s (4.5ms)
- But the output is **semantically useless**

#### Why It Fails: INT8 Quantization Destroys Embeddings

INT8 quantization with 30K vocab × 384 dims creates massive collisions:
- All tokens map to nearly identical embeddings after quantization
- CLS token output is identical across completely different inputs
- The model "runs" but produces garbage

#### Workarounds Attempted

1. **Attention mask removal** — Sub, Equal, Where ops are unsupported. Removed mask subgraph entirely, replaced with constant zeros. Compilation succeeds but doesn't fix the INT8 problem.

2. **FP16 embeddings** via `trigger_float_op`:
   ```
   disable & <{.*Gather.*}:float16_preferred!> <{.*embeddings.*}:float16_preferred!>
   ```
   CixBuilder supports per-node regex for mixed precision, but Z3 hardware doesn't support FP16 compute natively.

3. **TPC kernel duplicate symbols** — Multiple transformer encoder layers produce duplicate symbols in the TPC kernels. Fix: pass `--allow-multiple-definition` to `aipuold` (LLD-based linker), or patch ELF `.bin` files with alias symbols.

#### Verdict

BERT-style models **cannot** produce meaningful embeddings through INT8-only quantization without:
- Quantization-Aware Training (QAT)
- The full Compass SDK toolchain (not publicly available)
- Native FP16/FP32 compute support on the NPU

**Recommendation:** Use CLIP text encoder for text embeddings on the Z3 NPU. It works correctly and produces usable 256-dim vectors at 64 inf/s.

## Key Constraints

| Constraint | Detail |
|------------|--------|
| Quantization | INT8 only (no FP16/FP32 compute on Z3) |
| Target | Must use `X2_1204`, not `Z3` variants |
| Unsupported ops | Sub, Equal, Where (attention masks) |
| MatMul | Maps to "Convolution" — requires Convolution TPC kernel |
| Duplicate symbols | Multi-layer transformers need `--allow-multiple-definition` |
| Input layout | CHW (channels-first), not HWC — undocumented requirement |

## References

- [CIX AI Model Hub](https://modelscope.cn/models/cix/ai_model_hub_25_Q3) — pre-compiled models
- NPU driver: `orangepi-xunlong/component_cix-current` on GitHub
- TPC kernels source: `Orangepi6plus_1.0.2_debian_bookworm` vendor image → `ZhouyiOperators_x2` wheel
