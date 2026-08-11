#!/usr/bin/env python3
import argparse, json, re
from collections import defaultdict
from pathlib import Path

P = re.compile(r'tensor\[(\d+)\]: name = ([^,]+), size = (\d+), offset = (\d+)$')


def category(name):
    if name.startswith('blk.52.') or 'nextn' in name:
        return 'mtp'
    if re.search(r'ffn_(up|down)_exps', name):
        return 'routed_experts'
    if '_shexp' in name:
        return 'shared_expert'
    if 'ffn_gate_inp' in name or 'exp_probs' in name:
        return 'router'
    if '.ssm_' in name:
        return 'mamba'
    if re.search(r'\.attn_(q|k|v|output)\.', name):
        return 'attention'
    if name in ('token_embd.weight', 'output.weight'):
        return 'embedding_output'
    if 'norm' in name:
        return 'normalization'
    return 'other'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('header'); ap.add_argument('output')
    a=ap.parse_args(); tensors=[]
    for line in Path(a.header).read_text().splitlines():
        if line.startswith('gguf_ex_read_1:'): break
        m=P.search(line)
        if m: tensors.append({'index':int(m.group(1)), 'name':m.group(2), 'bytes':int(m.group(3)), 'offset':int(m.group(4))})
    cats=defaultdict(int); blocks=defaultdict(lambda:defaultdict(int))
    for t in tensors:
        c=category(t['name']); cats[c]+=t['bytes']
        m=re.match(r'blk\.(\d+)\.',t['name'])
        if m: blocks[m.group(1)][c]+=t['bytes']
    result={'tensor_count':len(tensors), 'total_tensor_bytes':sum(x['bytes'] for x in tensors),
            'categories_bytes':dict(sorted(cats.items())), 'blocks':{k:dict(v) for k,v in sorted(blocks.items(), key=lambda x:int(x[0]))},
            'tensors':tensors}
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'tensor_count':result['tensor_count'],'total_gib':result['total_tensor_bytes']/2**30,
                      'categories_gib':{k:round(v/2**30,4) for k,v in cats.items()}},indent=2))

if __name__=='__main__': main()
