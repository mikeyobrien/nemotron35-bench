#!/usr/bin/env python3
import argparse, json, re


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('inventory')
    ap.add_argument('--gpu-expert-blocks',type=int,required=True)
    ap.add_argument('--order',choices=['early','late','middle','alternating'],default='early')
    ap.add_argument('--include-mtp',action='store_true')
    a=ap.parse_args()
    d=json.load(open(a.inventory))
    blocks=[int(b) for b,c in d['blocks'].items() if int(b)<52 and c.get('routed_experts',0)]
    if a.order=='early': order=blocks
    elif a.order=='late': order=list(reversed(blocks))
    elif a.order=='middle': order=sorted(blocks,key=lambda b:abs(b-26))
    else: order=blocks[::2]+blocks[1::2]
    gpu=set(order[:a.gpu_expert_blocks]); cpu=[b for b in blocks if b not in gpu]
    if a.include_mtp: cpu.append(52)
    pat='^blk\\.('+'|'.join(map(str,cpu))+')\\.ffn_(up|down)_exps\\.weight$'
    print(json.dumps({'order':a.order,'gpu_expert_blocks':sorted(gpu),'cpu_expert_blocks':sorted(cpu),
                      'override':f'{pat}=CPU','server_args':['-ngl','all','-fit','off','-ot',f'{pat}=CPU']},indent=2))
if __name__=='__main__': main()
