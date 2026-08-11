#!/usr/bin/env python3
import argparse, json, os, signal, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path('/home/mobrienv/code/nemotron35-bench')
LLAMA = Path('/home/mobrienv/src/llama.cpp')
SERVER = LLAMA / 'build-cuvk/bin/llama-server'
PROBE = ROOT / 'scripts/probe.py'
MONITOR = ROOT / 'scripts/monitor.py'


def wait_health(port, timeout=360, process=None):
    end = time.monotonic() + timeout
    last = None
    while time.monotonic() < end:
        if process is not None and process.poll() is not None:
            print(f'server exited before health check (code {process.returncode})', file=sys.stderr)
            return False
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as r:
                if r.status == 200: return True
        except Exception as e: last = e
        time.sleep(2)
    print(f'health timeout: {last}', file=sys.stderr)
    return False


def wait_vram_free(limit=1600, timeout=60):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            x = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'], text=True)
            if int(x.strip()) < limit: return True
        except Exception: pass
        time.sleep(1)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('model')
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--workloads', default='short,coding')
    p.add_argument('--thinking', default='off')
    p.add_argument('--tools', action='store_true')
    p.add_argument('server_args', nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.server_args and args.server_args[0] == '--': args.server_args = args.server_args[1:]

    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'results' / f'{stamp}_{args.name}'
    out.mkdir(parents=True)
    subprocess.run(['pkill','-f','/llama-server'], check=False)
    wait_vram_free()

    commit = subprocess.check_output(['git','-C',str(LLAMA),'rev-parse','HEAD'], text=True).strip()
    cmd = [str(SERVER), '-m', str(Path(args.model).resolve()), '--port', str(args.port), '--no-webui', '--jinja', *args.server_args]
    receipt = {
        'date': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'name': args.name, 'commit': commit,
        'model': str(Path(args.model).resolve()), 'command': cmd, 'reps': args.reps,
        'workloads': args.workloads.split(','), 'thinking': args.thinking, 'tools': args.tools,
    }
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2))
    log = (out/'server.log').open('w')
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    mon = subprocess.Popen([sys.executable, str(MONITOR), str(srv.pid), str(out/'monitor.csv')])
    summaries = []
    try:
        if not wait_health(args.port, process=srv):
            raise RuntimeError('server failed health check')
        specs = {'short': 256, 'coding': 512, 'long': 256, 'tool': 256}
        for workload in args.workloads.split(','):
            prompt = ROOT/'workloads'/f'{workload}.txt'
            for rep in range(1, args.reps+1):
                raw = out/f'{workload}-{args.thinking}-r{rep}.json'
                pcmd = [sys.executable, str(PROBE), str(prompt), '--port', str(args.port), '--max-tokens', str(specs[workload]),
                        '--thinking', args.thinking, '--nonce', f'{args.name}-{workload}-{rep}', '--output', str(raw)]
                if args.tools or workload == 'tool': pcmd.append('--tools')
                proc = subprocess.run(pcmd, text=True, capture_output=True, timeout=2400)
                (out/f'{workload}-{args.thinking}-r{rep}.stderr').write_text(proc.stderr)
                if proc.returncode != 0:
                    summaries.append({'workload':workload,'rep':rep,'error':proc.stderr,'returncode':proc.returncode})
                else:
                    item=json.loads(proc.stdout); item.update({'workload':workload,'rep':rep}); summaries.append(item)
        (out/'summaries.json').write_text(json.dumps(summaries, indent=2))
        grouped={}
        for x in summaries:
            if 'decode_tps' not in x: continue
            grouped.setdefault(x['workload'],[]).append(x)
        medians={}
        for w,xs in grouped.items():
            medians[w]={k:statistics.median([x[k] for x in xs if x.get(k) is not None]) for k in ['decode_tps','prompt_tps','prompt_ms','wall_ms']}
        (out/'medians.json').write_text(json.dumps(medians, indent=2))
        print(json.dumps({'result_dir':str(out),'medians':medians}, indent=2))
    finally:
        if srv.poll() is None:
            os.killpg(srv.pid, signal.SIGTERM)
            try: srv.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(srv.pid, signal.SIGKILL)
        try: mon.wait(timeout=5)
        except subprocess.TimeoutExpired: mon.terminate()
        log.close()


if __name__ == '__main__':
    main()
