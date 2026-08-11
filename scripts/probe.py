#!/usr/bin/env python3
import argparse, json, statistics, time, urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--thinking", choices=["on", "off"], default="off")
    p.add_argument("--tools", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--nonce", default="")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    prompt = Path(args.prompt).read_text()
    if args.nonce:
        prompt = f"Benchmark nonce: {args.nonce}\n\n{prompt}"
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": args.thinking == "on", "force_nonempty_content": True},
    }
    if args.tools:
        body["tools"] = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the repository",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Repository-relative path"},
                        "line_start": {"type": "integer"},
                        "line_end": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            },
        }]
        body["tool_choice"] = "auto"

    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        raw = resp.read()
    wall_ms = (time.monotonic() - start) * 1000
    obj = json.loads(raw)
    Path(args.output).write_bytes(raw)

    timings = obj.get("timings", {})
    choice = obj.get("choices", [{}])[0]
    message = choice.get("message", {})
    predicted_ms = timings.get("predicted_ms") or 0
    predicted_n = timings.get("predicted_n") or 0
    summary = {
        "prompt": str(Path(args.prompt).resolve()),
        "thinking": args.thinking,
        "tools": args.tools,
        "wall_ms": round(wall_ms, 2),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_tokens": timings.get("prompt_n"),
        "prompt_tps": timings.get("prompt_per_second"),
        "completion_tokens": predicted_n,
        "decode_tps": (predicted_n * 1000 / predicted_ms) if predicted_ms else timings.get("predicted_per_second"),
        "finish_reason": choice.get("finish_reason"),
        "content_chars": len(message.get("content") or ""),
        "reasoning_chars": len(message.get("reasoning_content") or ""),
        "tool_calls": message.get("tool_calls") or [],
        "response_file": str(Path(args.output).resolve()),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
