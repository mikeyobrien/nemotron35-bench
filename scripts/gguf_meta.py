#!/usr/bin/env python3
import argparse, json, struct

T={0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',8:'str',9:'array',10:'u64',11:'i64',12:'f64'}
FMT={0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
SZ={k:struct.calcsize(v) for k,v in FMT.items()}

def readn(f,n):
    b=f.read(n)
    if len(b)!=n: raise EOFError(f'wanted {n}, got {len(b)} at {f.tell()}')
    return b

def u64(f): return struct.unpack('<Q',readn(f,8))[0]
def string(f): return readn(f,u64(f)).decode('utf-8','replace')

def scalar(f,t):
    if t==8:return string(f)
    return struct.unpack(FMT[t],readn(f,SZ[t]))[0]

def array(f, collect=True):
    et=struct.unpack('<I',readn(f,4))[0]; n=u64(f)
    if not collect:
        if et in SZ: f.seek(SZ[et]*n,1); return {'type':T.get(et,et),'count':n}
        if et==8:
            for _ in range(n): f.seek(u64(f),1)
            return {'type':'str','count':n}
    vals=[scalar(f,et) for _ in range(n)]
    return vals

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('file'); ap.add_argument('--include-large',action='store_true')
    a=ap.parse_args()
    with open(a.file,'rb') as f:
        magic=readn(f,4); ver=struct.unpack('<I',readn(f,4))[0]; nt=u64(f); nk=u64(f)
        meta={}
        for _ in range(nk):
            key=string(f); typ=struct.unpack('<I',readn(f,4))[0]
            if typ==9: val=array(f,a.include_large)
            else: val=scalar(f,typ)
            meta[key]=val
    print(json.dumps({'magic':magic.decode('ascii','replace'),'version':ver,'tensor_count':nt,'metadata_count':nk,'metadata':meta},indent=2))
if __name__=='__main__': main()
