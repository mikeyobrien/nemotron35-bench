#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mobrienv/code/nemotron35-bench
LLAMA=/home/mobrienv/src/llama.cpp/build-cuvk/bin/llama-server
MODEL="$ROOT/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf"
PORT="${PORT:-8080}"
CTX="${CTX:-65536}"
ALIAS="${ALIAS:-nemotron35-lightning-30b-a3b-nvfp4}"

# Eight routed-expert blocks remain on the RTX (1,3,6,8,10,13,15,17).
# All later routed-expert matrices stay in host RAM. Dense/Mamba/attention/
# shared-expert tensors remain on CUDA. Keep fit disabled so placement is stable.
exec "$LLAMA" \
  -m "$MODEL" --alias "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" --no-webui --jinja \
  --device CUDA0 -ngl all -fit off \
  -t 12 -tb 12 -c "$CTX" -np 1 \
  -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 \
  -ot '^blk\.(20|22|24|27|29|31|34|36|38|40|43|45|47|49|51)\.ffn_(up|down)_exps\.weight$=CPU'
