# Nemotron 3.5 Lightning 30B-A3B — local inference research

Measured recipe for running **NVIDIA-Nemotron-3.5-Lightning-30B-A3B** (hybrid
Mamba-2 + MoE + sparse Attention, 30B total / ~3B active) on a consumer laptop:

- Ryzen AI 9 HX 370 (30 GiB RAM)
- RTX 5070 Ti Mobile 12 GB (Blackwell, sm_120)
- Radeon 890M iGPU (unused — see RESULTS.md)
- CachyOS Linux

The focus is **real coding-agent and tool-use workloads** (repeated turns, long
prompts, tool schemas), not synthetic benchmark theater.

## Headline result

Use the **NVFP4 GGUF** with llama.cpp CUDA build, explicit tensor placement:
dense/Mamba/Attention/shared-expert/router tensors plus the first **eight**
routed-expert blocks on CUDA; the remaining 15 routed-expert blocks mmap'd in
RAM and computed on CPU in parallel. 12 physical CPU threads, one slot, Q8 KV,
flash attention, 64K context.

| Metric (16K coding-agent turn, median of 3) | NVFP4 | Q4_K_S |
|---|---:|---:|
| End-to-end | **43.26 s** | 49.15 s |
| Prefill | **528.9 t/s** | 423.4 t/s |
| Decode | 40.7 t/s | **45.5 t/s** |
| Peak VRAM | **8,990 MiB** | 10,712 MiB |

Q4_K_S decodes 11.6% faster but prefills ~20% slower — and prefill dominates
coding-agent turns. NVFP4 wins end-to-end by 13.6% with more memory headroom.

## Quick start

```bash
# 1. Build llama.cpp (CUDA, sm_120) — see RESULTS.md for the exact cmake line
# 2. Download the model
wget -O models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf \
  https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/resolve/main/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf

# 3. Run the recommended server
./start-nemotron.sh
```

The server exposes an OpenAI-compatible API on `127.0.0.1:8080`; a systemd user
unit template is included (see `start-nemotron.sh` header).

## Layout

- `scripts/` — benchmark harness (`run_experiment.py`, `probe.py`, `monitor.py`, `sweep.py`, tensor inventory, tool probes)
- `workloads/` — exact prompts (short review, 16K coding-agent context, tool selection, long context)
- `configs/` — baseline/finalist/selective placement configs
- `results/` — one immutable directory per experiment, with per-rep JSON + monitor traces
- `manifests/` — hardware, llama.cpp commit, model checksums, tensor inventories (provenance receipts)
- `RESEARCH.md` — upstream/model research receipts (vLLM/SGLang/TRT-LLM feasibility priors)
- `RESULTS.md` — the full ranked report: builds, commands, placements, all measured tables, rejected options

## Reproducibility

Every claimed number in `RESULTS.md` carries: date, llama.cpp commit
(`ebb546b`), build flags, exact command, exact model file + SHA-256, context and
batch settings, placement, environment notes, raw result directory, and median
values. See the quantization and speculative-decoding comparisons for why
alternatives (MTP, DFlash/DSpark, Radeon overflow, Q4_K_M/Q5) were rejected.

## License

Research artifacts and scripts in this repo are MIT. Model weights are NOT
included — they are distributed by NVIDIA under the OpenMDW-1.1 license via
Hugging Face (links in `RESEARCH.md`).
