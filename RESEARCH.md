# Upstream/model research receipts

Date: 2026-08-11

## Verified local upstream state

- llama.cpp checkout: `ebb546b` (`version 258`), newer than DFlash merge `cc078b45b635b3a59aa9adc1b888150ab67798a8`.
- Built with CUDA + Vulkan + `GGML_CUDA_ARCHITECTURES=120`.
- Devices: CUDA0 RTX 5070 Ti Laptop; Vulkan0 same NVIDIA GPU; Vulkan1 Radeon 890M RADV.
- Current server exposes `draft-mtp`, `draft-dflash`, and `draft-dspark`; separate draft tensor/device/CPU-MoE controls exist.
- Current DFlash code injects extracted target features into a persistent draft KV cache (`common/speculative.cpp`), rather than rebuilding the entire feature history each token.

## Model artifacts

- `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` is the preferred llama.cpp provenance.
- Its native NVFP4 GGUF contains the trained MTP block: 510 tensors, `block_count=53`, `nextn_predict_layers=1`, and 1,048,576-token GGUF context metadata.
- Bartowski Q4_K_S was converted before MTP export support: 401 tensors, `block_count=52`, no MTP tensors, despite a generic README statement that the architecture includes MTP.
- Q4_K_S tensor payload: 20.899 GiB, of which routed experts are 19.217 GiB and all non-routed tensors only 1.681 GiB.
- Native NVFP4 + MTP tensor payload: 20.910 GiB. Target routed experts are 15.389 GiB; MTP is 2.487 GiB; target non-routed tensors are about 3.035 GiB.
- NVFP4 routed-expert tensors are 685.1 MiB per target MoE block. The MTP MoE block alone is about 2.38 GiB.

## Context discrepancy

- NVIDIA BF16 `config.json` reports `max_position_embeddings=262144`.
- NVIDIA's model card advertises up to 1M context.
- llama.cpp's Granite hybrid converter deliberately writes `2**20` as context length for this non-finetuned RoPE hybrid path (`conversion/granite.py`).
- Both inspected GGUFs therefore report 1,048,576. This proves loader metadata, not quality/correctness at 1M; sustained testing is still required.

## Runtime feasibility priors

- vLLM 0.27.1 exposes selective CPU weight offload (`--cpu-offload-params`) and recognizes exact parameter-name segments such as `experts`, but NVIDIA's published recipe assumes DGX Spark/H100-class memory.
- SGLang exposes grouped CPU layer offload, not the same static GGUF tensor override granularity.
- TensorRT-LLM's official recipe assumes the full checkpoint fits the accelerator and is therefore not a practical first path on 12 GB.

## Primary sources

- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
- https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF
- https://github.com/ggml-org/llama.cpp/pull/26905
- local llama.cpp source at commit `ebb546b`
