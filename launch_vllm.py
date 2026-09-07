"""
launch_vllm.py -- start a vLLM OpenAI-compatible server and WAIT until it is up.

Fixes over the original:
  * model name is a CLI arg, so it can't drift from what eval.py requests
    (the original served Qwen3-0.6B while eval.py asked for Qwen3-1.7B; vLLM
    rejects the mismatched model name)
  * 0.6B is the MULTIMODAL problem statement's dev model. This PS is
    1.7B dev / 4B final. Default set accordingly.
  * polls /v1/models instead of sleep(30); 30s is not enough to load 4B

    python launch_vllm.py --model Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request


def wait_until_ready(port: int, timeout: int = 900) -> bool:
    url = f"http://localhost:{port}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    print(f"\nready after {time.time() - t0:.0f}s")
                    return True
        except (urllib.error.URLError, OSError):
            pass
        print(".", end="", flush=True)
        time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", a.model,
        "--port", str(a.port),
        "--gpu-memory-utilization", str(a.gpu_mem),
        "--max-model-len", str(a.max_len),
        "--enable-prefix-caching",
    ]
    print("launching:", " ".join(cmd))
    with open("vllm.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    print(f"serving {a.model} on :{a.port}, logs -> vllm.log")
    if wait_until_ready(a.port, a.timeout):
        print(f"pid {proc.pid}. stop with: kill {proc.pid}")
    else:
        print(f"\nnot ready after {a.timeout}s -- check vllm.log")
        proc.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
