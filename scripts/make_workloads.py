#!/usr/bin/env python3
from pathlib import Path

ROOT = Path('/home/mobrienv/code/nemotron35-bench')
SRC = Path('/home/mobrienv/src/llama.cpp')
OUT = ROOT/'workloads'; OUT.mkdir(parents=True, exist_ok=True)
files = [
    'src/models/nemotron-h.cpp', 'common/speculative.cpp', 'src/llama-model.cpp',
    'src/llama-context.cpp', 'common/arg.cpp', 'ggml/src/ggml-cuda/mmq.cu',
]
texts=[]
for rel in files:
    text=(SRC/rel).read_text(errors='replace')
    texts.append(f'\n===== FILE: {rel} =====\n{text}\n')
alltext=''.join(texts)
short_ctx=texts[0][:3000]
(OUT/'short.txt').write_text('Review this code excerpt. Explain which tensors are on the decode critical path and identify one likely local-inference bottleneck.\n'+short_ctx)
(OUT/'coding.txt').write_text('''You are investigating Nemotron-H hybrid inference performance in llama.cpp. Review the supplied repository context. Produce a concrete, technically justified plan to keep latency-sensitive Mamba, attention, router, and shared-expert tensors on a 12GB CUDA GPU while routed experts overflow to CPU RAM. Cite exact tensor names and code paths from the context, identify correctness risks, and propose the next three controlled benchmarks. Do not invent APIs.\n'''+alltext[:56000])
(OUT/'long.txt').write_text('''Analyze this large llama.cpp repository snapshot as a coding agent. Trace how Nemotron-H target inference and MTP/DFlash speculative inference load tensors, build state, and execute decode. Identify likely long-context scaling bottlenecks, graph-reuse risks, and the smallest upstream-compatible measurements needed to distinguish them. Cite filenames and functions.\n'''+alltext[:145000])
(OUT/'tool.txt').write_text('''Use the read_file tool to inspect src/models/nemotron-h.cpp lines 260 through 310 before answering. After the tool result, explain how routed and shared expert outputs are combined. Do not answer from memory; make the tool call first.''')
print({p.name: p.stat().st_size for p in OUT.glob('*.txt')})
