#!/usr/bin/env python3
import argparse,json,time,urllib.request
from pathlib import Path

def post(port,messages,tools,thinking):
 p={'model':'model','messages':messages,'tools':tools,'tool_choice':'auto','temperature':0,'seed':42,'max_tokens':256,'chat_template_kwargs':{'enable_thinking':thinking}}
 req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(p).encode(),headers={'Content-Type':'application/json'})
 t=time.perf_counter()
 with urllib.request.urlopen(req,timeout=1200) as r:o=json.load(r)
 return o,round((time.perf_counter()-t)*1000,3)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8080);ap.add_argument('--thinking',choices=['on','off'],default='off');ap.add_argument('--output',required=True);a=ap.parse_args()
 tools=[{'type':'function','function':{'name':'read_counter','description':'Read an integer counter from a named service','parameters':{'type':'object','properties':{'service':{'type':'string'}},'required':['service']}}}]
 msgs=[{'role':'system','content':'You are a coding agent. Use tools when required. Never invent tool results.'},{'role':'user','content':'Call read_counter for service "build-queue". After receiving the result, report the integer and say whether it is odd or even.'}]
 first,ms1=post(a.port,msgs,tools,a.thinking=='on'); m=first['choices'][0]['message']; calls=m.get('tool_calls') or []
 valid_call=len(calls)==1 and calls[0].get('function',{}).get('name')=='read_counter'
 valid_args=False
 if valid_call:
  try: valid_args=json.loads(calls[0]['function']['arguments']).get('service')=='build-queue'
  except Exception: pass
 second=None;ms2=None;valid_final=False
 if valid_call:
  msgs += [m,{'role':'tool','tool_call_id':calls[0]['id'],'content':'{"value":17}'}]
  second,ms2=post(a.port,msgs,tools,a.thinking=='on'); text=(second['choices'][0]['message'].get('content') or '').lower(); valid_final='17' in text and 'odd' in text
 out={'thinking':a.thinking,'first_ms':ms1,'second_ms':ms2,'valid_call':valid_call,'valid_args':valid_args,'valid_final':valid_final,'first':first,'second':second}
 Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ('first','second')},indent=2))
if __name__=='__main__':main()
