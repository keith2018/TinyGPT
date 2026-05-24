#!/usr/bin/env python3
"""
OpenAI-compatible serving benchmark for TinyGPT.

Heavily inspired by vLLM's `benchmarks/benchmark_serving.py` — we hit the same
`/v1/completions` and `/v1/chat/completions` endpoints and compute the same
metric suite (TTFT / TPOT / ITL / E2E / throughput) so results can be compared
side-by-side with a vLLM server hosting the same model.

============================================================================
Starting the servers
============================================================================

TinyGPT server
--------------
Build the server target (once), then launch it pointing at a local HF-format
model directory. Port 8080 is the default; pick any free port.

    # build
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j --target TinyGPT_server

    # run (pick any free port; we use 8080 below)
    ./server/bin/TinyGPT_server \\
        --model /path/to/Qwen2.5-1.5B \\
        --host 0.0.0.0 --port 8080

    # verify: should list the loaded model
    curl -s http://localhost:8080/v1/models | jq .

vLLM server
-----------
Install vLLM into its own env (CUDA required). Match the same `--model` path
and `--dtype` as TinyGPT so the comparison is apples-to-apples, and use a
DIFFERENT port so both services can run side-by-side.

    pip install vllm       # or: uv pip install vllm
    python -m vllm.entrypoints.openai.api_server \\
        --model /path/to/Qwen2.5-1.5B \\
        --dtype bfloat16 \\
        --host 0.0.0.0 --port 8081 \\
        --max-model-len 4096 \\
        --disable-log-requests

    # verify
    curl -s http://localhost:8081/v1/models | jq .

Notes:
  * If you want to match TinyGPT's default paged-KV settings, pass
    `--block-size 16 --gpu-memory-utilization 0.85` to vLLM.
  * vLLM requires `stream_options: {include_usage: true}` in SSE requests to
    emit usage in the final chunk. This script sets that automatically, so no
    extra flag is needed on the vLLM side.

============================================================================
Running the benchmark (same command, different --base-url)
============================================================================

# Random synthetic prompts, Poisson arrivals at 8 req/s, streaming:
python benches/benchmark_serving.py \\
    --base-url http://localhost:8080 \\
    --model Qwen2.5-1.5B --tokenizer /path/to/Qwen2.5-1.5B \\
    --dataset random --num-prompts 500 --request-rate 8 \\
    --random-input-len 512 --random-output-len 128 \\
    --save-result tinygpt_random_r8.json

# ShareGPT dataset (download ShareGPT_V3_unfiltered_cleaned_split.json first):
python benches/benchmark_serving.py \\
    --base-url http://localhost:8080 \\
    --model Qwen2.5-1.5B --tokenizer /path/to/Qwen2.5-1.5B \\
    --dataset sharegpt --sharegpt-path /path/ShareGPT_V3.json \\
    --num-prompts 1000 --request-rate inf \\
    --save-result tinygpt_sharegpt.json

# Same thing against vLLM — only --base-url changes:
python benches/benchmark_serving.py \\
    --base-url http://localhost:8081 \\
    --model Qwen2.5-1.5B --tokenizer /path/to/Qwen2.5-1.5B \\
    --dataset sharegpt --sharegpt-path /path/ShareGPT_V3.json \\
    --num-prompts 1000 --request-rate inf \\
    --save-result vllm_sharegpt.json

Tips for a fair comparison:
  * Use `--model` = the id reported by `/v1/models` on each server (vLLM
    typically reports the full path; TinyGPT reports the last path segment).
  * Fix `--seed` so both runs see the same prompts and arrival jitter.
  * `--request-rate inf` stresses raw throughput; set a finite rate (e.g. 2,
    4, 8, 16) to measure latency under controlled load (gives a latency/QPS
    curve like vLLM's blog posts).
  * Warm both servers first (`--warmup 5` is the default for this script).
  * If you only care about decode throughput, set `--random-input-len 32`
    and a large `--random-output-len`; for prefill-heavy runs, invert it.

Dependencies:
  pip install aiohttp
  # Optional but recommended:
  pip install numpy transformers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable

try:
    import aiohttp
except ImportError:
    sys.exit(
        "Missing dependency: install with `pip install aiohttp numpy` (numpy is optional)"
    )

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # percentiles fall back to a manual path

try:
    from transformers import AutoTokenizer  # type: ignore
except ImportError:
    AutoTokenizer = None  # random dataset still works without it


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RequestSample:
    prompt: str
    prompt_len: int  # in tokens (approximate if tokenizer unavailable)
    output_len: int  # requested max_tokens


@dataclass
class RequestResult:
    success: bool = False
    prompt_len: int = 0
    output_len: int = 0  # actually generated tokens
    generated_text: str = ""
    ttft: float = 0.0  # seconds
    itl: list[float] = field(default_factory=list)
    latency: float = 0.0  # end-to-end, seconds
    error: str = ""


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def sample_random_dataset(
        num_prompts: int,
        input_len: int,
        output_len: int,
        range_ratio: float,
        tokenizer: Any | None,
        seed: int,
) -> list[RequestSample]:
    """Build random prompts via repeated token id sampling.

    If a tokenizer is available we produce token sequences of the desired
    length and decode them; otherwise we fall back to length-approximated
    English-ish text (1 token ≈ 4 chars).
    """
    rng = random.Random(seed)
    out: list[RequestSample] = []
    lo = max(1, int(input_len * (1 - range_ratio)))
    hi = max(lo, int(input_len * (1 + range_ratio)))

    if tokenizer is not None:
        vocab_size = tokenizer.vocab_size
        for _ in range(num_prompts):
            plen = rng.randint(lo, hi)
            ids = [rng.randint(0, vocab_size - 1) for _ in range(plen)]
            prompt = tokenizer.decode(ids, skip_special_tokens=True)
            # recount after detokenize (may differ slightly)
            try:
                plen = len(tokenizer.encode(prompt, add_special_tokens=False))
            except Exception:
                pass
            out.append(RequestSample(prompt=prompt, prompt_len=plen, output_len=output_len))
    else:
        # ~4 chars per token as a rough proxy
        for _ in range(num_prompts):
            plen = rng.randint(lo, hi)
            prompt = "hi " * plen
            out.append(RequestSample(prompt=prompt, prompt_len=plen, output_len=output_len))
    return out


SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered"
    "/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
SHAREGPT_DEFAULT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"


def ensure_sharegpt_dataset(path: str | None) -> str:
    """Return a local path to the ShareGPT V3 JSON, downloading on demand.

    * `path=None`  → cache at `$XDG_CACHE_HOME/tinygpt/<file>` (default
      `~/.cache/tinygpt/<file>`).
    * `path` is an existing directory → download into that directory.
    * `path` is an existing file      → use as-is.
    * `path` points to a non-existent file → download to that exact path.
    """
    if path is None:
        cache_dir = os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "tinygpt",
        )
        path = os.path.join(cache_dir, SHAREGPT_DEFAULT_FILENAME)
    elif os.path.isdir(path):
        path = os.path.join(path, SHAREGPT_DEFAULT_FILENAME)

    if os.path.exists(path):
        return path

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    print(
        f"ShareGPT dataset not found at {path}; downloading from {SHAREGPT_URL} ...",
        file=sys.stderr,
    )
    import urllib.request

    tmp_path = path + ".tmp"
    try:
        with urllib.request.urlopen(SHAREGPT_URL) as resp, open(tmp_path, "wb") as f:
            total_hdr = resp.headers.get("Content-Length")
            total = int(total_hdr) if total_hdr else None
            chunk_size = 1 << 20  # 1 MiB
            downloaded = 0
            last_report = 0.0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.perf_counter()
                if now - last_report >= 0.5:
                    if total:
                        pct = 100.0 * downloaded / total
                        print(
                            f"  downloaded {downloaded / 1e6:7.1f} / "
                            f"{total / 1e6:.1f} MB ({pct:5.1f}%)",
                            end="\r",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"  downloaded {downloaded / 1e6:7.1f} MB",
                            end="\r",
                            file=sys.stderr,
                        )
                    last_report = now
        os.replace(tmp_path, path)
        print(f"\nSaved ShareGPT dataset to {path}", file=sys.stderr)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    return path


def sample_sharegpt_dataset(
        path: str,
        num_prompts: int,
        tokenizer: Any | None,
        max_input_len: int,
        fixed_output_len: int | None,
        seed: int,
) -> list[RequestSample]:
    """Load ShareGPT-format JSON and pick (prompt, completion) pairs.

    ShareGPT layout (v3): list of conversations, each with a `conversations`
    list of {"from": "human"|"gpt", "value": str} turns.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pairs: list[tuple[str, str]] = []
    for conv in raw:
        turns = conv.get("conversations") or conv.get("turns") or []
        # Walk pairs of (human, gpt)
        for i in range(len(turns) - 1):
            a, b = turns[i], turns[i + 1]
            if a.get("from") == "human" and b.get("from") == "gpt":
                pairs.append((a.get("value", ""), b.get("value", "")))

    rng = random.Random(seed)
    rng.shuffle(pairs)

    out: list[RequestSample] = []
    for prompt, completion in pairs:
        if not prompt or not completion:
            continue
        if tokenizer is not None:
            try:
                plen = len(tokenizer.encode(prompt, add_special_tokens=False))
                clen = len(tokenizer.encode(completion, add_special_tokens=False))
            except Exception:
                plen = max(1, len(prompt) // 4)
                clen = max(1, len(completion) // 4)
        else:
            plen = max(1, len(prompt) // 4)
            clen = max(1, len(completion) // 4)

        if plen < 4 or clen < 4:
            continue
        if plen > max_input_len:
            continue

        out.append(
            RequestSample(
                prompt=prompt,
                prompt_len=plen,
                output_len=fixed_output_len if fixed_output_len else clen,
            )
        )
        if len(out) >= num_prompts:
            break

    if len(out) < num_prompts:
        print(
            f"WARN: requested {num_prompts} prompts, only {len(out)} fit constraints",
            file=sys.stderr,
        )
    return out


# ---------------------------------------------------------------------------
# Request pacing
# ---------------------------------------------------------------------------


async def pace_requests(
        samples: list[RequestSample], request_rate: float, burstiness: float
) -> AsyncIterator[RequestSample]:
    """Yield samples at the requested arrival rate.

    * `request_rate = inf` → fire all at once (max concurrency test).
    * `burstiness = 1.0` → Poisson (exponential inter-arrivals, vLLM default).
    * Other burstiness values use a Gamma(k=burstiness, theta=1/(rate*burstiness))
      distribution, matching vLLM's implementation so the comparison is fair.
    """
    if request_rate == float("inf"):
        for s in samples:
            yield s
        return

    assert burstiness > 0.0
    theta = 1.0 / (request_rate * burstiness)
    for s in samples:
        yield s
        # Gamma(k=burstiness, theta) — reduces to Exp(rate) when burstiness=1
        interval = random.gammavariate(burstiness, theta)
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Request execution
# ---------------------------------------------------------------------------


async def send_request_completions(
        session: aiohttp.ClientSession,
        base_url: str,
        model: str,
        sample: RequestSample,
        stream: bool,
        extra_body: dict[str, Any],
        tokenizer: Any | None,
) -> RequestResult:
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": sample.prompt,
        "max_tokens": sample.output_len,
        "temperature": 0.0,
        "stream": stream,
        # Ask servers that support it to include usage in the final SSE chunk.
        # vLLM requires this; TinyGPT always includes it — harmless either way.
        "stream_options": {"include_usage": True} if stream else None,
        **extra_body,
    }
    # drop None keys for cleanliness
    payload = {k: v for k, v in payload.items() if v is not None}

    result = RequestResult(prompt_len=sample.prompt_len, output_len=0)
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:200]}"
                return result

            if not stream:
                data = await resp.json()
                t1 = time.perf_counter()
                result.latency = t1 - t0
                result.ttft = t1 - t0  # no token-level timing in non-stream mode
                choice = data.get("choices", [{}])[0]
                result.generated_text = choice.get("text", "")
                usage = data.get("usage") or {}
                result.output_len = usage.get("completion_tokens") or _count_tokens(
                    result.generated_text, tokenizer
                )
                result.success = True
                return result

            # ---- streaming ----
            last_t = t0
            first_token_seen = False
            generated = []
            completion_tokens_from_usage: int | None = None
            async for line in resp.content:
                line = line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload_s = line[len("data:"):].strip()
                if payload_s == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload_s)
                except json.JSONDecodeError:
                    continue
                # token delta
                delta = ""
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("text", "") or (
                            choices[0].get("delta", {}) or {}
                    ).get("content", "")
                if delta:
                    now = time.perf_counter()
                    if not first_token_seen:
                        result.ttft = now - t0
                        first_token_seen = True
                    else:
                        result.itl.append(now - last_t)
                    last_t = now
                    generated.append(delta)
                # final chunk with usage
                if chunk.get("usage"):
                    completion_tokens_from_usage = chunk["usage"].get("completion_tokens")

            t_end = time.perf_counter()
            result.latency = t_end - t0
            result.generated_text = "".join(generated)
            if completion_tokens_from_usage is not None:
                result.output_len = completion_tokens_from_usage
            else:
                result.output_len = _count_tokens(result.generated_text, tokenizer)
            result.success = first_token_seen and result.output_len > 0
            if not first_token_seen:
                result.error = "no tokens received"
            return result
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        return result


def _count_tokens(text: str, tokenizer: Any | None) -> int:
    if not text:
        return 0
    if tokenizer is None:
        return max(1, len(text) // 4)
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    if np is not None:
        return float(np.percentile(xs, p))
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs_sorted[int(k)]
    return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)


def summarize(results: list[RequestResult], wall_time: float) -> dict[str, Any]:
    ok = [r for r in results if r.success]
    ttfts = [r.ttft for r in ok if r.ttft > 0]
    itls = [x for r in ok for x in r.itl]
    e2es = [r.latency for r in ok]
    # TPOT = (e2e - ttft) / max(1, output_len-1) per request
    tpots = []
    for r in ok:
        if r.output_len > 1 and r.latency > r.ttft:
            tpots.append((r.latency - r.ttft) / (r.output_len - 1))

    total_in = sum(r.prompt_len for r in ok)
    total_out = sum(r.output_len for r in ok)

    summary = {
        "completed": len(ok),
        "failed": len(results) - len(ok),
        "wall_time_s": wall_time,
        "request_throughput": len(ok) / wall_time if wall_time > 0 else 0.0,
        "output_throughput": total_out / wall_time if wall_time > 0 else 0.0,
        "total_token_throughput": (total_in + total_out) / wall_time if wall_time > 0 else 0.0,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
    }
    for name, xs in (("ttft_ms", [x * 1000 for x in ttfts]),
                     ("tpot_ms", [x * 1000 for x in tpots]),
                     ("itl_ms", [x * 1000 for x in itls]),
                     ("e2e_ms", [x * 1000 for x in e2es])):
        if xs:
            summary[f"{name}_mean"] = statistics.fmean(xs)
            summary[f"{name}_median"] = statistics.median(xs)
            summary[f"{name}_p90"] = _pct(xs, 90)
            summary[f"{name}_p99"] = _pct(xs, 99)
        else:
            summary[f"{name}_mean"] = float("nan")
    return summary


def print_summary(s: dict[str, Any], request_rate: float | str, burstiness: float) -> None:
    def row(label, value, unit=""):
        print(f"  {label:<28}{value:>14} {unit}")

    print("\n=============== Serving benchmark summary ===============")
    row("Request rate",
        "inf" if request_rate == float("inf") else f"{request_rate:.3g}",
        "req/s")
    row("Burstiness", f"{burstiness:.2f}", "")
    row("Successful requests", s["completed"])
    row("Failed requests", s["failed"])
    row("Benchmark duration", f"{s['wall_time_s']:.2f}", "s")
    row("Total input tokens", s["total_input_tokens"])
    row("Total generated tokens", s["total_output_tokens"])
    row("Request throughput", f"{s['request_throughput']:.3f}", "req/s")
    row("Output token throughput", f"{s['output_throughput']:.2f}", "tok/s")
    row("Total token throughput", f"{s['total_token_throughput']:.2f}", "tok/s")

    print("  --- Time to First Token ---")
    row("mean / median", f"{s.get('ttft_ms_mean', float('nan')):.2f} / "
                         f"{s.get('ttft_ms_median', float('nan')):.2f}", "ms")
    row("p90 / p99", f"{s.get('ttft_ms_p90', float('nan')):.2f} / "
                     f"{s.get('ttft_ms_p99', float('nan')):.2f}", "ms")

    print("  --- Time Per Output Token (excl. 1st) ---")
    row("mean / median", f"{s.get('tpot_ms_mean', float('nan')):.2f} / "
                         f"{s.get('tpot_ms_median', float('nan')):.2f}", "ms")
    row("p90 / p99", f"{s.get('tpot_ms_p90', float('nan')):.2f} / "
                     f"{s.get('tpot_ms_p99', float('nan')):.2f}", "ms")

    print("  --- Inter-Token Latency ---")
    row("mean / median", f"{s.get('itl_ms_mean', float('nan')):.2f} / "
                         f"{s.get('itl_ms_median', float('nan')):.2f}", "ms")
    row("p90 / p99", f"{s.get('itl_ms_p90', float('nan')):.2f} / "
                     f"{s.get('itl_ms_p99', float('nan')):.2f}", "ms")

    print("  --- End-to-End Latency ---")
    row("mean / median", f"{s.get('e2e_ms_mean', float('nan')):.2f} / "
                         f"{s.get('e2e_ms_median', float('nan')):.2f}", "ms")
    row("p90 / p99", f"{s.get('e2e_ms_p90', float('nan')):.2f} / "
                     f"{s.get('e2e_ms_p99', float('nan')):.2f}", "ms")
    print("==========================================================\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    # ---- Tokenizer ----
    tokenizer = None
    if args.tokenizer:
        if AutoTokenizer is None:
            print("WARN: transformers not installed, skipping tokenizer", file=sys.stderr)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer, trust_remote_code=args.trust_remote_code
            )

    # ---- Dataset ----
    if args.dataset == "random":
        samples = sample_random_dataset(
            num_prompts=args.num_prompts,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            seed=args.seed,
        )
    elif args.dataset == "sharegpt":
        sharegpt_path = ensure_sharegpt_dataset(args.sharegpt_path)
        samples = sample_sharegpt_dataset(
            path=sharegpt_path,
            num_prompts=args.num_prompts,
            tokenizer=tokenizer,
            max_input_len=args.sharegpt_max_input_len,
            fixed_output_len=args.sharegpt_output_len,
            seed=args.seed,
        )
    else:
        sys.exit(f"Unknown dataset: {args.dataset}")

    if not samples:
        sys.exit("no samples produced — check your dataset args")

    request_rate: float = float("inf") if args.request_rate == "inf" else float(args.request_rate)

    # ---- Warmup ----
    conn = aiohttp.TCPConnector(limit=args.max_concurrency or 0, force_close=False)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        if args.warmup > 0:
            print(f"Warming up with {args.warmup} requests...", file=sys.stderr)
            warm_samples = samples[: args.warmup]
            warm_tasks = [
                send_request_completions(session, args.base_url, args.model, s,
                                         stream=args.stream, extra_body={}, tokenizer=tokenizer)
                for s in warm_samples
            ]
            await asyncio.gather(*warm_tasks)

        # ---- Main run ----
        pending: list[asyncio.Task[RequestResult]] = []
        results: list[RequestResult] = []
        sem = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
        print(f"Sending {len(samples)} requests at rate={request_rate} req/s ...", file=sys.stderr)

        async def _go(s: RequestSample) -> RequestResult:
            if sem is not None:
                async with sem:
                    return await send_request_completions(
                        session, args.base_url, args.model, s, args.stream, {}, tokenizer
                    )
            return await send_request_completions(
                session, args.base_url, args.model, s, args.stream, {}, tokenizer
            )

        t_start = time.perf_counter()
        async for s in pace_requests(samples, request_rate, args.burstiness):
            pending.append(asyncio.create_task(_go(s)))
        # Progress log as results land
        completed = 0
        for fut in asyncio.as_completed(pending):
            r = await fut
            results.append(r)
            completed += 1
            if completed % max(1, len(pending) // 20) == 0:
                print(f"  [{completed}/{len(pending)}] done", file=sys.stderr)
        t_end = time.perf_counter()

    summary = summarize(results, t_end - t_start)
    print_summary(summary, request_rate, args.burstiness)

    if args.save_result:
        out = {
            "args": {k: v for k, v in vars(args).items() if not callable(v)},
            "summary": summary,
            "per_request": [vars(r) for r in results],
        }
        with open(args.save_result, "w", encoding="utf-8") as f:
            json.dump(out, f, default=str, indent=2)
        print(f"Wrote detailed results to {args.save_result}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8080",
                   help="OpenAI-compatible server base URL")
    p.add_argument("--model", required=True,
                   help="Model id as reported by /v1/models on the server")
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer id/path for length bookkeeping; "
                        "usually the same repo as --model")
    p.add_argument("--trust-remote-code", action="store_true")

    # dataset
    p.add_argument("--dataset", choices=["random", "sharegpt"], default="random")
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    # random
    p.add_argument("--random-input-len", type=int, default=256)
    p.add_argument("--random-output-len", type=int, default=128)
    p.add_argument("--random-range-ratio", type=float, default=0.0,
                   help="Uniform ±ratio on input length (0=fixed)")
    # sharegpt
    p.add_argument("--sharegpt-path", type=str, default=None,
                   help="Path to ShareGPT_V3_unfiltered_cleaned_split.json, or a "
                        "directory to cache it in. If unset or the file does not "
                        "exist, it is downloaded to ~/.cache/tinygpt/ automatically.")
    p.add_argument("--sharegpt-max-input-len", type=int, default=2048)
    p.add_argument("--sharegpt-output-len", type=int, default=None,
                   help="If set, override dataset output with fixed max_tokens")

    # load
    p.add_argument("--request-rate", default="inf",
                   help="Requests per second, or 'inf' for unlimited concurrency")
    p.add_argument("--burstiness", type=float, default=1.0,
                   help="Arrival distribution shape (1.0 = Poisson, matches vLLM)")
    p.add_argument("--max-concurrency", type=int, default=0,
                   help="Cap in-flight requests (0 = no cap)")
    p.add_argument("--stream", action="store_true", default=True,
                   help="Use SSE streaming to measure TTFT/ITL (default on)")
    p.add_argument("--no-stream", dest="stream", action="store_false")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timeout", type=float, default=600.0)

    # output
    p.add_argument("--save-result", type=str, default=None,
                   help="Dump detailed JSON results to this path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
