#!/usr/bin/env python3
"""Measure steady-state RoboLab serving latency with one recorded DROID observation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import websockets.asyncio.client

from flux_action.serving import protocol


def summarize(values: list[float]) -> dict[str, float]:
    sample = np.asarray(values, dtype=np.float64)
    return {
        "count": int(sample.size),
        "mean_ms": float(sample.mean()),
        "std_ms": float(sample.std()),
        "min_ms": float(sample.min()),
        "p50_ms": float(np.percentile(sample, 50)),
        "p90_ms": float(np.percentile(sample, 90)),
        "p95_ms": float(np.percentile(sample, 95)),
        "p99_ms": float(np.percentile(sample, 99)),
        "max_ms": float(sample.max()),
    }


def make_request(observation: Path, prompt: str) -> dict:
    with np.load(observation, allow_pickle=False) as arrays:
        state = arrays["state"].astype(np.float32, copy=True)
        return {
            "observation/wrist_image_left": arrays["images.wrist"].copy(),
            "observation/exterior_image_1_left": arrays["images.left"].copy(),
            "observation/exterior_image_2_left": arrays["images.right"].copy(),
            "observation/joint_position": state[:-1],
            "observation/gripper_position": state[-1:],
            "prompt": prompt,
        }


async def benchmark(endpoint: str, request: dict, warmup: int, iterations: int) -> dict:
    packer = protocol.Packer()
    rtt_ms: list[float] = []
    infer_ms: list[float] = []
    action_shape = None
    async with websockets.asyncio.client.connect(
        endpoint, compression=None, max_size=None, ping_timeout=None
    ) as websocket:
        metadata = protocol.unpackb(await websocket.recv())
        for index in range(warmup + iterations):
            started = time.perf_counter()
            await websocket.send(packer.pack(request))
            response = protocol.unpackb(await websocket.recv())
            elapsed_ms = (time.perf_counter() - started) * 1000
            if isinstance(response, str):
                raise RuntimeError(response)
            actions = np.asarray(response["action"])
            if not np.isfinite(actions).all():
                raise RuntimeError("server returned non-finite actions")
            action_shape = list(actions.shape)
            if index >= warmup:
                rtt_ms.append(elapsed_ms)
                infer_ms.append(float(response["server_timing"]["infer_ms"]))
    return {
        "server_metadata": metadata,
        "action_shape": action_shape,
        "warmup_requests": warmup,
        "measured_requests": iterations,
        "server_inference": summarize(infer_ms),
        "client_rtt": summarize(rtt_ms),
        "steady_requests_per_second": 1000.0 / statistics.mean(infer_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations must be positive")

    request = make_request(args.observation, args.prompt)
    result = asyncio.run(benchmark(args.endpoint, request, args.warmup, args.iterations))
    result.update(
        {
            "endpoint": args.endpoint,
            "observation": str(args.observation),
            "observation_sha256": hashlib.sha256(args.observation.read_bytes()).hexdigest(),
            "prompt": args.prompt,
            "runtime": {
                "host": platform.node(),
                "python": platform.python_version(),
            },
        }
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
