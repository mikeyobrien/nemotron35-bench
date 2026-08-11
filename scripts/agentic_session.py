#!/usr/bin/env python3
import argparse,json,time,urllib.request
from pathlib import Path

def post(port,messages,max_tokens,thinking):
 p={'model':'model','messages':messages,'temperature':0.2,'top_p':0.95,'seed':42,'max_tokens':max_tokens,'chat_template_kwargs':{'enable_thinking':thinking}}
 req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(p).encode(),headers={'Content-Type':'application/json'})
 t=time.perf_counter()
 with urllib.request.urlopen(req,timeout=3600) as r:o=json.load(r)
 return o,round((time.perf_counter()-t)*1000,3)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8080);ap.add_argument('--seconds',type=int,default=900);ap.add_argument('--thinking',choices=['on','off'],default='off');ap.add_argument('--output',required=True);a=ap.parse_args()
 context=Path('/home/mobrienv/code/nemotron35-bench/workloads/coding.txt').read_text()
 messages=[{'role':'system','content':'You are a precise coding agent. Cite functions from the supplied repository context and never invent APIs.'},{'role':'user','content':context}]
 followups=['Now rank those benchmarks by expected information gain and justify the ranking.','Turn the top benchmark into an exact llama-server command using only flags already mentioned in the context.','Review your prior answer for any invented flag or unsupported tensor name and correct it.','Explain how you would validate tensor placement from server logs.','Summarize the current best plan in five terse bullets.']
 start=time.monotonic(); rows=[]; i=0
 while time.monotonic()-start<a.seconds:
  o,ms=post(a.port,messages,256,a.thinking=='on'); msg=o['choices'][0]['message']; messages.append(msg); tim=o.get('timings',{}); rows.append({'turn':i,'wall_ms':ms,'prompt_n':tim.get('prompt_n'),'prompt_ms':tim.get('prompt_ms'),'predicted_n':tim.get('predicted_n'),'predicted_ms':tim.get('predicted_ms'),'predicted_per_second':tim.get('predicted_per_second'),'finish_reason':o['choices'][0].get('finish_reason'),'content_chars':len(msg.get('content') or ''),'reasoning_chars':len(msg.get('reasoning_content') or '')})
  messages.append({'role':'user','content':followups[i%len(followups)]}); i+=1
 Path(a.output).write_text(json.dumps({'seconds_target':a.seconds,'elapsed_s':round(time.monotonic()-start,3),'thinking':a.thinking,'turns':rows,'final_messages':messages},indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
