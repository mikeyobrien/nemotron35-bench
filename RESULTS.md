# Nemotron 3.5 Lightning 30B-A3B — HX 370 + RTX 5070 Ti results

Date: 2026-08-11  
Machine: Ryzen AI 9 HX 370, 30 GiB RAM, Radeon 890M (RADV), RTX 5070 Ti Mobile 12 GB, CachyOS  
llama.cpp: `ebb546b7e961bd46fd9ed0387ffd14ca86b6fe1b` (version 258)  
Target: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf`  
Size: `22,459,679,904` bytes  
SHA-256: `7827805ae9f2d20cc71e46bf05d9cb045e222d3fa0429363c324bbf6d3cab959`

## Executive conclusion

Use the ggml-org NVFP4 GGUF with CUDA for all dense, Mamba, attention, shared-expert, router, and the first eight routed-expert blocks. Keep the remaining routed-expert matrices in CPU-mapped RAM. Use 12 physical CPU threads, `-fit off`, one slot, Q8 KV, flash attention, and a 64K context for daily agent work.

This is the best measured **end-to-end coding-agent** configuration, not the highest isolated decode number. Auto-fit produced slightly higher raw decode on some short runs, but explicit early-8 placement had better cold coding-turn completion time, more VRAM headroom, reproducible placement, and better sustained behavior.

Recommended launcher: [`start-nemotron.sh`](./start-nemotron.sh).

## Exact build

```bash
git -C /home/mobrienv/src/llama.cpp merge --ff-only origin/master
cmake -S /home/mobrienv/src/llama.cpp \
  -B /home/mobrienv/src/llama.cpp/build-cuvk -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_VULKAN=ON \
  -DGGML_CUDA_ARCHITECTURES=120 \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
ninja -C /home/mobrienv/src/llama.cpp/build-cuvk \
  llama-server llama-bench llama-cli llama-gguf
```

## Recommended server command

```bash
/home/mobrienv/src/llama.cpp/build-cuvk/bin/llama-server \
  -m /home/mobrienv/code/nemotron35-bench/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf \
  --host 127.0.0.1 --port 8080 --no-webui --jinja \
  --device CUDA0 -ngl all -fit off \
  -t 12 -tb 12 -c 262144 -np 4 \
  -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 \
  -ot '^blk\.(24|27|29|31|34|36|38|40|43|45|47|49|51)\.ffn_(up|down)_exps\.weight$=CPU'
```

RTX routed-expert blocks: `1, 3, 6, 8, 10, 13, 15, 17, 20, 22` (early10).  
CPU routed-expert blocks: `24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51`.

Do not omit `-fit off`; otherwise auto-fit can silently change placement. Do not set `VK_DRIVER_FILES=radeon_icd.x86_64.json`. The recommended command does not use the Radeon.

## Final 64K agent workload: three cold/non-LCP repetitions

A unique prefix prevented large prompt-prefix reuse between repetitions. Values are medians.

| Workload | Prompt tokens | Output tokens | Prompt t/s | Decode t/s | End-to-end |
|---|---:|---:|---:|---:|---:|
| Short code review | 1,073–1,078 | 256 | 407.3 | 42.83 | 8.75 s |
| Coding-agent context | 15,970 | 512 | 528.9 | 40.73 | 43.26 s |
| Tool selection | 80 new / 388 cold | 57 | 101.4 | 43.01 | 2.22 s |

All three tool-selection repetitions returned one correctly structured `read_file` call with the requested path and line range. Raw directory:

`results/20260811_130316_final_early8_64k/`

Observed peaks over the run:

- VRAM: 8,990 MiB.
- GPU power: 45.45 W.
- GPU temperature: 76 C.
- Host memory available never fell below 17.85 GiB.
- Process swap peaked at 0.212 GiB; no destructive swapping occurred.

## Sustained growing agent session

A five-minute, one-slot, growing-message session at 128K context completed 39 consecutive turns without a crash or malformed response.

- Elapsed: 306.1 s.
- First turn: 15,980-token prompt, 256-token output, 39.50 s end-to-end, 40.00 decode t/s.
- Median later turn: 7.24 s.
- Median decode: 36.99 t/s.
- First-five decode mean: 37.69 t/s.
- Last-five decode mean: 36.55 t/s.
- Peak VRAM: 9,258 MiB.
- Peak temperature: 78 C.
- Minimum available host RAM: 18.47 GiB.
- Process swap peak: 0.150 GiB.

The small sustained decline is consistent with longer active state and thermal/context overhead; it did not become unstable. Raw directory:

`results/agentic_early8_128k_20260811_125309/`

## Quantization comparison: NVFP4 versus Q4_K_S

Repository artifact sizes were checked directly against the Hugging Face tree:

| Candidate | Repository size | Practical decision |
|---|---:|---|
| Q3_K_L | 19,498,225,600 bytes | Only smaller high-quality K-quant with enough size reduction to plausibly change RTX residency; dedicated benchmark queued |
| Q4_K_S | 22,447,520,704 bytes | Downloaded and benchmarked |
| NVFP4 | 22,459,679,904 bytes | Downloaded and benchmarked; current winner |
| Q4_K_M | 24,725,740,480 bytes | Rejected: 2.28 GB larger than Q4_K_S, which already used up to 1.41 GiB process swap |
| Q5_K_S | 24,008,614,336 bytes | Rejected for the same 30 GiB host-headroom constraint |
| Q5_K_M | 26,212,516,288 bytes | Rejected: leaves inadequate OS, graph, cache, and agent headroom |

The completed bartowski Q4_K_S artifact was tested after the initial report rather than inferred from size:

- File: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_S.gguf`
- Size: `22,447,520,704` bytes
- SHA-256: `d1dd036281235dbdf1acc370634d209f6c3412f576bad2773d1e24a9d35f9d51`
- Workload: three cold repetitions each of short, 15,973-token coding, and tool-selection requests; thinking off; 64K context; one slot; 12 threads; Q8 KV; flash attention.

| Quant / placement | Coding prefill | Coding decode | Coding wall | Short wall | Tool wall | Peak VRAM | Process swap |
|---|---:|---:|---:|---:|---:|---:|---:|
| **NVFP4 explicit early-8** | **528.87 t/s** | 40.73 t/s | **43.26 s** | **8.75 s** | 2.22 s | **8,990 MiB** | **0.21 GiB** |
| Q4_K_S explicit early-8 | 412.55 t/s | 43.81 t/s | 50.50 s | 9.19 s | 2.40 s | 9,918 MiB | 1.41 GiB |
| Q4_K_S auto-fit, 1 GiB target | 423.41 t/s | **45.46 t/s** | 49.15 s | 9.17 s | **2.05 s** | 10,712 MiB | 0.96 GiB |

Q4_K_S auto-fit decoded 11.6% faster than NVFP4 on the coding request, but prefilling was 19.9% slower and the complete coding-agent turn was **13.6% slower**. It also consumed 1.72 GiB more peak VRAM and about 0.75 GiB more process swap. Q4_K_S won the tiny tool-selection request by 7.4%, where its decode advantage outweighed the small prompt, but that is below the primary coding-agent objective. Both Q4 configurations produced correct structured tool calls in all three repetitions.

**Quant verdict:** NVFP4 remains the daily-use winner. Blackwell NVFP4 materially improves the long-prefill portion that dominates realistic coding-agent turns, while preserving more memory headroom. Q4_K_S is a valid decode-oriented alternative, not the fastest end-to-end agent configuration.

Raw directories:

- `results/20260811_135744_final_q4ks_early8_64k/`
- `results/20260811_140243_final_q4ks_autofit_64k/`

## Placement screening

All entries used the same NVFP4 target, 32K context, batch 2048/512, Q8 KV, flash attention, one slot, and 12 threads unless the row states otherwise.

| Configuration | Short decode t/s | Prompt t/s | End-to-end | Verdict |
|---|---:|---:|---:|---|
| Auto-fit, 1 GiB target headroom | 49.82 | 207.1 | 10.32 s | Highest simple decode; placement is implicit |
| Explicit early 8 expert blocks | 46.49 | 238.0 | 10.00 s | Best overall after coding tests |
| Explicit late 8 | 47.75 | 251.6 | 9.62 s | Strong short result; slower coding cold turn |
| Explicit middle 8 | 46.57 | 267.1 | 9.51 s | Strong prefill; slower coding cold turn |
| Alternating 8 | 47.45 | 281.6 | 9.21 s | Best short E2E; decode fell on coding workload |
| Explicit early 6 | 43.06 | 268.8 | 9.94 s | Too little expert residency |
| Explicit early 4 | 37.84 | 170.6 | 13.04 s | Too little expert residency |
| CPU-MoE, 10 threads | 35.23 | 202.2 | 12.55 s | Useful fallback, not fastest |
| CPU-MoE, 12 threads | 34.48 | 224.0 | 12.20 s | Similar to 10 threads |
| CPU-MoE, 24 SMT threads | 8.09 | 162.7 | 38.21 s | Catastrophic SMT bandwidth contention |
| Naive `-ngl 28` | 19.81 | 349.5 | 16.00 s | Poor decode despite good prefill |
| Naive `-ngl 20` | 16.00 | 196.9 | 21.44 s | Reject |
| CPU-only | 9.38 | 117.6 | 36.38 s | Baseline only |

Compared with the directly comparable screening baselines, explicit early-8 decode was 190.5% faster than naive `-ngl 20` and 395.8% faster than CPU-only.

Cold 15,960-token coding-turn results before the final nonce harness:

| Placement | Prompt t/s | Decode t/s | End-to-end |
|---|---:|---:|---:|
| Early 8 | 501.0 | 39.81 | 44.80 s |
| Alternating 8 | 516.1 | 36.84 | 44.90 s |
| Middle 8 | 493.3 | 40.53 | 45.06 s |
| Late 8 | 477.2 | 41.40 | 45.91 s |
| Auto-fit | 454.9 | 46.27 | 46.23 s |

Early-8 finished the coding turn 3.1% sooner than auto-fit even though auto-fit decoded faster. This is why auto-fit is not the recommendation.

Raw screening receipts:

- `results/sweep-20260811_123602.json`
- `results/sweep-20260811_123854.json`
- `results/sweep-20260811_124530.json`

## Context verdict

The official metadata is inconsistent across checkpoints: BF16-derived configuration reports 262,144 while NVIDIA NVFP4 metadata reports 1,048,576. Treat 1M as a checkpoint capability claim, not a practical setting on this laptop.

| Context | Result | Peak VRAM | Evidence |
|---:|---|---:|---|
| 64K | Recommended daily setting; 39,363-token prompt completed | 8,990 MiB | `results/20260811_124913_long_64k_early8/` |
| 128K | Loaded and ran; recommended when extra conversation room matters | 9,258 MiB | `results/20260811_125108_ctx_128k_early8/` |
| 256K | Loaded and ran with explicit placement | 9,794 MiB | `results/20260811_125144_ctx_256k_early8/` |
| 1M | Failed during context creation: CUDA OOM allocating 2,189,254,784-byte compute buffer | — | `results/20260811_125221_ctx_1m_early8/server.log` |

The 39,363-token long workload processed at 525.65 prompt t/s and 36.27 decode t/s, completing in 82.07 s. Only the 64K run exercised a genuinely long prompt; the 128K/256K runs establish allocation and basic inference, not 128K/256K retrieval quality.

64K was 4.9% faster end-to-end and 10.9% faster in decode than the final 128K coding configuration, so 64K is the speed/capacity sweet spot. Use 128K only when the agent actually needs it.

## Reasoning and tool calling

- `enable_thinking=false`: recommended for routine coding-agent loops and tool selection.
- `enable_thinking=true`: structurally correct `reasoning_content` plus tool call was produced, but a 256-token cap truncated a code-review answer inside reasoning. Give thinking turns at least 512–1024 output tokens.
- The bundled Jinja template correctly emitted OpenAI-compatible `tool_calls`.
- Recommended client values are in `agent-client-recommendation.json`.

Thinking/tool receipt: `results/20260811_125849_thinking_tool_early8/`.

## Speculative decoding

### Native MTP

The NVFP4 GGUF really does contain `blk.52.nextn.*` plus the trained MTP attention/MoE block. Base inference intentionally logs those tensors as unused. Current llama.cpp can create an embedded MTP draft context with `--spec-type draft-mtp`.

Measured CPU-draft MTP with early-8 target placement:

- Acceptance: 52.34% (`123 / 235`), mean draft length 2.56.
- Decode: 31.65 t/s.
- Comparable target-only early-8 screening: 46.49 t/s.
- MTP was 31.9% slower.

Moving the MTP layer to GPU forced target residency down to early-4 and was worse again: 28.63 t/s. MTP is therefore **not recommended** on 12 GB VRAM with current recurrent-state rollback/verification overhead.

Raw receipts:

- `results/20260811_124621_mtp_early8_cpu/`
- `results/20260811_124705_mtp_early4_gpu/`

### DFlash and DSpark

Neither NVIDIA sidecar is distributed as a ready-to-run GGUF. Current primary-source state also leaves meaningful caveats:

- DFlash requires converting an approximately 1.18 GB sidecar and finding another VRAM/RAM placement budget.
- DSpark requires an approximately 1.35 GB sidecar; its sliding-window metadata path is affected by the still-open llama.cpp fix tracked in PR #26900.
- The target already leaves only a few GiB of useful VRAM headroom; MTP showed that an extra drafter can lose rather than gain speed under partial offload.

No compatible ready-made GGUF was found, so neither path was promoted to a daily configuration or assigned fabricated performance numbers.

## Radeon 890M verdict

One bounded RADV/Vulkan run succeeded on current master, so the old `MUL_MAT_ID` crash is not universal anymore. It was still uncompetitive:

- Early four expert blocks on CUDA, remaining expert matrices on `Vulkan1`: 24.83 t/s.
- Same early-four shape with overflow in CPU RAM: 37.84 t/s.
- Radeon overflow was 34.4% slower.

The iGPU shares system-memory bandwidth with the CPU and adds cross-backend scheduling. Keep it disabled for this model. Receipt: `results/20260811_124820_radeon_overflow_early4/`.

## Alternate runtime feasibility

| Runtime | Practical verdict on this 12 GB GeForce + 30 GiB host |
|---|---|
| llama.cpp | Supported, measured, stable; only practical winner |
| vLLM | Official day-zero recipe exists, but target exceeds VRAM and SM120 NVFP4 + CPU weight offload has a reported garbage-output bug (vLLM issue #38718); reject for daily use |
| SGLang | Day-zero model support exists, but hybrid linear-attention/Mamba CPU weight offload still depends on open fixes such as PR #23474; validated hardware list does not include this GeForce laptop |
| TensorRT-LLM | Nemotron-H/NVFP4 source support exists, but the official supported-hardware list names datacenter Blackwell/Hopper/Ampere/Ada parts, not RTX 5070 Ti; no suitable verified host-overflow path |
| Ollama | Installed, but wraps llama.cpp while hiding the exact tensor placement that produces the win; convenience option only |
| LM Studio | Same GGUF backend class, less reproducible placement control; no performance reason to prefer it |

Pulling an 8.48 GiB compressed vLLM container was intentionally not treated as a benchmark: the required CPU-offload path is known-bad on SM120 for this model class and could not produce trustworthy output.

## Comparison with Qwen3.6-35B-A3B

Prior measured Qwen3.6 early-14 selective placement reached 63.9 t/s on the coding workload and about 64 t/s generally. Nemotron's final 64K coding median was 40.73 t/s, 36.3% slower. Nemotron prompt processing was competitive (about 529 t/s), but its active top-6 routed-expert accesses plus hybrid state/verification path remain much more expensive per generated token on this placement.

For this machine, Qwen3.6 remains the throughput winner. Choose Nemotron 3.5 for its model behavior/tool quality, not for local decode speed.

## Bottleneck diagnosis

The primary bottleneck is routed-expert weight movement/host-memory bandwidth, not arithmetic throughput:

- Naive layer splitting was dramatically worse than expert-only placement.
- 24 SMT threads collapsed decode from the mid-30s to 8.1 t/s.
- Radeon shared-memory overflow was slower than CPU overflow.
- Sustained GPU utilization averaged only 46.2%, with roughly 45 W peak power, while decode remained in the high-30s.
- Placement of eight expert blocks changed prefill/decode tradeoffs, but no fixed early/middle/late region dominated every workload.

Static hot-expert caching is not currently exposed as a production llama.cpp feature. The modest differences among early/middle/late/alternating eight-block placements do not justify a custom cache implementation without first collecting per-expert routing traces. The concrete next experiment, if one is pursued, is routing-trace collection plus an offline cache simulation—not an unmeasured patch to daily inference.

## Reproducibility map

- Machine receipt: `manifests/machine.txt`
- Build receipt: `manifests/build-command.txt`
- Commit: `manifests/llama-commit.txt`
- Model checksum: `manifests/nvfp4-sha256.txt`
- Q4_K_S checksum: `manifests/q4_k_s-sha256.txt`
- Tensor inventory: `manifests/nvfp4-tensor-inventory.json`
- GGUF metadata: `manifests/nvfp4-meta.json`
- Research sources: `RESEARCH.md`
- Harness: `scripts/run_experiment.py`, `scripts/probe.py`, `scripts/monitor.py`, `scripts/agentic_session.py`
- Final launcher: `start-nemotron.sh`
- Q4_K_S comparison: `results/20260811_135744_final_q4ks_early8_64k/`, `results/20260811_140243_final_q4ks_autofit_64k/`
