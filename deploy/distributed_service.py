#!/usr/bin/env python3
"""Four-rank RoboLab service: two TP=2 replicas execute CFG branches concurrently."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from flux_action.policy import FluxActionPolicy
from flux_action.serving.robolab import (
    COMPOSITE_HW,
    observation_image,
    observation_state,
    serve,
)


class DistributedCoordinator:
    """Rank-zero adapter that dispatches each request to the three worker ranks."""

    def __init__(self, policy, control_group) -> None:
        self.policy = policy
        self.control_group = control_group
        self.queries = 0
        self.action_dim = policy.config.action_dim
        self.prompt = None
        self.prompt_id = 0
        self.control = torch.empty(3, dtype=torch.int64, device=policy.device)

    def infer(self, obs: dict[str, Any], *, seed: int | None = None) -> dict[str, Any]:
        image = observation_image(obs)
        state = observation_state(obs, self.action_dim)
        prompt = str(obs.get("prompt", "") or "")
        self.queries += 1
        query_seed = self.queries if seed is None else seed
        prompt_changed = prompt != self.prompt
        if prompt_changed:
            self.prompt = prompt
            self.prompt_id += 1
        self.control.copy_(torch.tensor((1, query_seed, self.prompt_id), device=self.policy.device))
        dist.broadcast(self.control, src=0)
        if prompt_changed:
            prompt_message = [prompt]
            dist.broadcast_object_list(prompt_message, src=0, group=self.control_group)
        image_gpu = torch.from_numpy(image).to(self.policy.device)
        state_gpu = torch.from_numpy(state).to(self.policy.device)
        dist.broadcast(image_gpu, src=0)
        dist.broadcast(state_gpu, src=0)
        actions = self.policy.predict_from_composite(
            image_gpu, state_gpu, prompt, seed=query_seed
        )
        actions = actions.detach().cpu().numpy().astype(np.float32)
        return {"action": actions}


def build_policy(checkpoint: Path, tp_group, branch: str, *, warmup: int):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    policy = FluxActionPolicy.from_pretrained(checkpoint, device=device)
    policy.prepare_inference(compile=False)
    policy.dit.enable_tensor_parallel(tp_group)
    policy.enable_parallel_cfg(branch, dist.group.WORLD, tensor_parallel_size=2)
    policy.reset()
    policy.dit.compile_static()
    policy.dit.compile_hot()
    policy.video_vae.module.compile_encoder()
    policy._inference_prepared = True
    policy.serving_setup = {
        "precision": "bfloat16",
        "compiled": True,
        "world_size": 4,
        "tensor_parallel_size": 2,
        "parallel_cfg_replicas": 2,
    }
    image = torch.zeros((*COMPOSITE_HW, 3), dtype=torch.uint8)
    state = torch.zeros(policy.config.action_dim, dtype=torch.float32)
    for index in range(warmup):
        policy.predict_from_composite(image, state, "warm-up request", seed=index)
        torch.cuda.synchronize()
        dist.barrier()
    policy.reset()
    return policy


def worker_loop(policy, control_group) -> None:
    image = torch.empty((*COMPOSITE_HW, 3), dtype=torch.uint8, device=policy.device)
    state = torch.empty(policy.config.action_dim, dtype=torch.float32, device=policy.device)
    control = torch.empty(3, dtype=torch.int64, device=policy.device)
    prompt = ""
    prompt_id = 0
    while True:
        dist.broadcast(control, src=0)
        kind, seed, next_prompt_id = (int(value) for value in control.cpu())
        if kind == 0:
            return
        if kind != 1:
            raise RuntimeError(f"unknown distributed command {kind!r}")
        if next_prompt_id != prompt_id:
            prompt_message = [None]
            dist.broadcast_object_list(prompt_message, src=0, group=control_group)
            prompt = prompt_message[0]
            prompt_id = next_prompt_id
        dist.broadcast(image, src=0)
        dist.broadcast(state, src=0)
        policy.predict_from_composite(image, state, prompt, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"distributed service requires exactly four ranks, got {world}")
    groups = (dist.new_group((0, 1)), dist.new_group((2, 3)))
    control_group = dist.new_group(backend="gloo")
    tp_group = groups[0] if rank < 2 else groups[1]
    branch = "positive" if rank < 2 else "negative"
    policy = build_policy(args.checkpoint, tp_group, branch, warmup=args.warmup)
    dist.barrier()
    if rank == 0:
        metadata = {"serving_setup": policy.serving_setup}
        print(
            json.dumps(
                {
                    "ready": True,
                    "host": os.uname().nodename,
                    "port": args.port,
                    **metadata,
                }
            ),
            flush=True,
        )
        serve(
            DistributedCoordinator(policy, control_group),
            host=args.host,
            port=args.port,
            metadata=metadata,
        )
    else:
        worker_loop(policy, control_group)


if __name__ == "__main__":
    main()
