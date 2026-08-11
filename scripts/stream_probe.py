#!/usr/bin/env python3
import argparse, json, statistics, time, urllib.request
from pathlib import Path

def pct(xs,p):
    if not xs:return None
    xs=sorted(xs); return xs[min(len(xs)-1,round((len(xs)-1)*p))]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('prompt'); ap.add_argument('--port',type=int,default=8080); ap.add_argument('--max-tokens',type=int,default=512); ap.add_argument('--thinking',choices=['on','off'],default='off'); ap.add_argument('--output',required=True)
    a=ap.parse_args(); prompt=Path(a.prompt).read_text()
    payload={'model':'model','messages':[{'role':'user','content':prompt}], 'temperature':1.0,'top_p':0.95,'seed':42,'max_tokens':a.max_tokens,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':a.thinking=='on'}}
    req=urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start=time.perf_counter(); first=None; times=[]; content=[]; reasoning=[]; usage={}; finish=None; events=[]
    with urllib.request.urlopen(req,timeout=3600) as r:
        for raw in r:
            line=raw.decode('utf-8','replace').strip()
            if not line.startswith('data: '): continue
            data=line[6:]
            if data=='[DONE]': break
            obj=json.loads(data); now=time.perf_counter(); events.append(obj)
            if obj.get('usage'): usage=obj['usage']
            for ch in obj.get('choices',[]):
                d=ch.get('delta',{}); text=d.get('content') or ''; thought=d.get('reasoning_content') or ''
                if text or thought:
                    if first is None:first=now
                    times.append(now); content.append(text); reasoning.append(thought)
                if ch.get('finish_reason'): finish=ch['finish_reason']
    end=time.perf_counter(); gaps=[(b-a)*1000 for a,b in zip(times,times[1:])]
    result={'prompt':str(Path(a.prompt).resolve()),'thinking':a.thinking,'wall_ms':round((end-start)*1000,3),'ttft_ms':round((first-start)*1000,3) if first else None,'event_count':len(times),'itlt_p50_ms':round(statistics.median(gaps),3) if gaps else None,'itlt_p95_ms':round(pct(gaps,.95),3) if gaps else None,'usage':usage,'finish_reason':finish,'content':' '.join(content),'reasoning_content':' '.join(reasoning),'raw_events':events}
    Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k not in ('content','reasoning_content','raw_events')},indent=2))
if __name__=='__main__':main()
