#!/usr/bin/env python3
"""Один и тот же запрос к yforge и llama.cpp: prefill и decode.

Пути задаются через окружение: MODEL, YFORGE_BIN, LLAMA_SERVER,
LLAMA_LIB_DIR, BENCH_LOG_DIR. Параметры конфигурации yforge берутся из
EXTRA (JSON), например:
  EXTRA='{"MOE_CPU_GPU_EXPERTS":"2"}' python3 ab_llama.py yforge 131072 130000
"""

import json, os, signal, socket, statistics, subprocess, sys, time
import urllib.request, urllib.error

MODEL = os.environ.get("MODEL", "/models/Ornith-1.5-35B-Q4_K_M.gguf")
YF = os.environ.get("YFORGE_BIN", "target/release/yforge")
LS = os.environ.get("LLAMA_SERVER", "llama-server")
LS_LIB = os.environ.get("LLAMA_LIB_DIR", "")


def vram_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def wait_gpu_free(limit=600, tries=160):
    for _ in range(tries):
        if vram_mib() <= limit:
            return True
        time.sleep(3)
    return False


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_ready(port, proc, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        if not port_free(port):
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models",
                                             headers={"Authorization": "Bearer x"})
                with urllib.request.urlopen(req, timeout=5):
                    pass
                return True
            except Exception:
                pass
        time.sleep(2)
    return False


def make_prompt(target_tokens):
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
             "hotel", "india", "juliet", "kilo", "lima", "mike", "november"]
    chunk = " ".join(words)
    n = max(1, int(target_tokens / 1.31 / len(words)))
    body = (chunk + " ") * n
    return ("Read the following journal entries and answer the final question.\n"
            + body +
            "\nQuestion: how many entries mention 'lima'? Answer with a number.")


def launch(engine, ctx, port, logp):
    env = dict(os.environ)
    log = open(logp, "wb")
    if engine == "yforge":
        env.update({
            "MODEL": MODEL, "CTX": str(ctx), "SLOTS": "1", "PORT": str(port),
            "API_KEYS": '[{"key":"x","name":"bench"}]',
            "QWEN36_HOST": "127.0.0.1", "QWEN36_PORT": str(port),
            "GPU_ONLY": "1", "GPU_LAYERS": "999",
            "CONTEXT_LIMIT": str(ctx), "GRAPH_WINDOW": str(ctx),
            "KV_POOL_Q8": "0", "PREFILL_CHUNK": "2048", "FLASH_ATTN": "0",
            "PGRAPH": "on", "MTP": "0", "PREFIX_CACHE_MIB": "0",
            "F16_FAST_ACC": "1", "VRAM_HEADROOM_MIB": "1024",
            "PREFILL_MIN_FREE_MIB": "512", "MOE_EXPERTS": "auto",
            "EXPERT_PROMOTE_PER_STEP": "64", "PREFILL_SM75": "1",
            "MOE_CPU": "1", "CUDA_GRAPHS": "1", "DECODE_GQA_MMA": "1",
            "REQ_TIMEOUT": "7200", "RUST_BACKTRACE": "1",
        })
        env.update(json.loads(os.environ.get("EXTRA", "{}")))
        nctl = os.environ.get("NCTL", "--cpunodebind=1 --membind=1").split()
        cmd = ([] if nctl == ["none"] else ["numactl", *nctl]) + [YF]
    else:
        if LS_LIB:
            env["LD_LIBRARY_PATH"] = LS_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
        cmd = ["numactl", "--cpunodebind=1", "--membind=1", LS, "-m", MODEL,
               "-c", str(ctx), "-ngl", "99", "-ot", "exps=CPU", "-t", "36",
               "--host", "127.0.0.1", "--port", str(port), "-fa", "1",
               "-np", "1", "-b", "2048", "-ub", "512", "--no-webui"]
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True), log


def one_request(port, prompt, max_tokens):
    body = json.dumps({
        "stream": True, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer x"})
    t0 = time.time()
    first = last = None
    usage = None
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta:
                    now = time.time()
                    if first is None:
                        first = now
                    last = now
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens")
    ttft = (first - t0) if first else None
    ds = (last - first) if (first and last) else None
    return {
        "prompt_tokens": pt, "completion_tokens": ct,
        "ttft_s": ttft,
        "prefill_tps": (pt / ttft) if (pt and ttft) else None,
        "decode_tps": ((ct - 1) / ds) if (ct and ds) else None,
    }


def run(engine, ctx, prompt_tokens, port=18311, reps=1):
    if not wait_gpu_free():
        return {"engine": engine, "ctx": ctx, "error": "GPU busy"}
    logp = os.environ.get("BENCH_LOG_DIR", "/tmp") + f"/bench-{engine}-{ctx}.log"
    proc, log = launch(engine, ctx, port, logp)
    try:
        if not wait_ready(port, proc):
            return {"engine": engine, "ctx": ctx, "error": "server not ready",
                    "log": logp}
        prompt = make_prompt(prompt_tokens)
        rows = [one_request(port, prompt, 64) for _ in range(reps)]
        pre = [r["prefill_tps"] for r in rows if r["prefill_tps"]]
        dec = [r["decode_tps"] for r in rows if r["decode_tps"]]
        return {
            "engine": engine, "ctx": ctx,
            "prompt_tokens": rows[0]["prompt_tokens"],
            "prefill_tps": round(statistics.median(pre), 1) if pre else None,
            "decode_tps": round(statistics.median(dec), 2) if dec else None,
            "runs": [{"prefill": round(r["prefill_tps"], 1) if r["prefill_tps"] else None,
                      "decode": round(r["decode_tps"], 2) if r["decode_tps"] else None}
                     for r in rows],
            "log": logp,
        }
    except urllib.error.HTTPError as e:
        return {"engine": engine, "ctx": ctx, "error": f"HTTP {e.code}",
                "detail": e.read().decode("utf-8", "ignore")[:400], "log": logp}
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=90)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        log.close()
        time.sleep(4)


if __name__ == "__main__":
    engine = sys.argv[1]
    ctx = int(sys.argv[2])
    pt = int(sys.argv[3])
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 18311
    reps = int(os.environ.get("REPS", "1"))
    print(json.dumps(run(engine, ctx, pt, port, reps), ensure_ascii=False))
