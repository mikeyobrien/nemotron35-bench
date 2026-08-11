#!/usr/bin/env python3
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT=Path('/home/mobrienv/code/nemotron35-bench')
RUN=ROOT/'scripts/run_experiment.py'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('configs'); ap.add_argument('model'); ap.add_argument('--reps',type=int,default=3); ap.add_argument('--workloads',default='short')
    a=ap.parse_args(); configs=json.load(open(a.configs)); results=[]
    for c in configs:
        cmd=[sys.executable,str(RUN),'--reps',str(a.reps),'--workloads',a.workloads,'--thinking',c.get('thinking','off'),c['name'],a.model,'--',*c['args']]
        print('RUN',json.dumps(cmd),flush=True); t=time.time()
        p=subprocess.run(cmd,text=True,capture_output=True,timeout=c.get('timeout',3600))
        item={'name':c['name'],'command':cmd,'returncode':p.returncode,'wall_s':round(time.time()-t,2),'stdout':p.stdout,'stderr':p.stderr}
        try:
            x=json.loads(p.stdout); item['result_dir']=x.get('result_dir'); item['medians']=x.get('medians')
        except Exception: pass
        results.append(item); print(json.dumps(item,indent=2),flush=True)
        (ROOT/'results'/'sweep-progress.json').write_text(json.dumps(results,indent=2))
    out=ROOT/'results'/f'sweep-{time.strftime("%Y%m%d_%H%M%S")}.json'; out.write_text(json.dumps(results,indent=2)); print(out)
if __name__=='__main__':main()
