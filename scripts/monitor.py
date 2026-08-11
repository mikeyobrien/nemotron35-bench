#!/usr/bin/env python3
import argparse, csv, os, subprocess, time
from pathlib import Path


def read_meminfo():
    out = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k, v = line.split(':', 1)
        out[k] = int(v.strip().split()[0])
    return out


def read_proc(pid):
    try:
        status = {}
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if ':' in line:
                k, v = line.split(':', 1)
                status[k] = v.strip()
        stat = Path(f'/proc/{pid}/stat').read_text().split()
        kb = lambda k: int(status.get(k, '0 kB').split()[0])
        return kb('VmRSS'), kb('VmSwap'), int(stat[9]), int(stat[11])
    except (FileNotFoundError, ProcessLookupError):
        return None


def nvidia():
    try:
        text = subprocess.check_output([
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem',
            '--format=csv,noheader,nounits'], text=True, timeout=2).strip()
        return [x.strip() for x in text.split(',')]
    except Exception:
        return ['', '', '', '', '', '']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pid', type=int)
    p.add_argument('output')
    p.add_argument('--interval', type=float, default=0.5)
    args = p.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    amd_busy_path = Path('/sys/class/drm/card2/device/gpu_busy_percent')
    fields = ['unix_s','gpu_util_pct','gpu_mem_mib','gpu_power_w','gpu_temp_c','gpu_sm_mhz','gpu_mem_mhz',
              'amd_busy_pct','mem_available_kib','swap_free_kib','proc_rss_kib','proc_swap_kib','proc_minflt','proc_majflt']
    with path.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(fields); f.flush()
        while True:
            proc = read_proc(args.pid)
            if proc is None: break
            mem = read_meminfo()
            amd = amd_busy_path.read_text().strip() if amd_busy_path.exists() else ''
            w.writerow([time.time(), *nvidia(), amd, mem.get('MemAvailable',''), mem.get('SwapFree',''), *proc])
            f.flush(); time.sleep(args.interval)


if __name__ == '__main__':
    main()
