# Copyright 2026 Black Forest Labs. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone FLUX Action policy with local checkpoint export and strict restoration.

The forward path is a local dense-batch implementation. Distributed token-budget
training, EMA and training-state resume are not implemented here.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .checkpoints.artifacts import sha256
from .config import ACTION_ONLY_CONTENT_STREAMS, REQUIRED_CONTENT_STREAMS, PolicyConfig
from .hub import resolve_policy_directory
from .inference import sampling
from .models.positional import batched_prc_action, batched_prc_vid, times_to_ids
from .models.runtime import TRUNK_WEIGHTS_FILENAME, resolve_weights
from .models.text_encoder import VEC_DIM, text_contexts
from .models.transformer import JointSingleSeq, JointSingleSeqParams
from .models.wiring import (
    action_dit_params,
    build_action_dit,
    fresh_module_names,
    restrict_content_streams,
)
from .processing import history, normalization, packing

ACTION = "action"
ACTION_PREV = "action_prev"  # the command before a training window (joint_delta parameterization)
OBS_STATE = "state"
GRAY_LEVEL = 128  # the flat tile a dropped camera turns into


def _released_inference_config(config: PolicyConfig) -> bool:
    return (
        config.torch_dtype == "bfloat16"
        and config.action_modality == "action_prediction_droid"
        and config.action_dim == 8
        and config.chunk_size == 32
        and tuple(config.content_streams or ()) == ("video", "video_cond")
        and JointSingleSeqParams(**{**config.dit_config, "attn_mode": config.attn_mode})
        == JointSingleSeqParams(attn_mode=config.attn_mode)
        and config.inference_profile == "default"
        and config.sampler == "cosmos_unipc"
    )


def _verify_policy_manifest(directory: Path, config_name: str) -> None:
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing {path.name}")
    manifest = json.loads(path.read_text())
    assert manifest.get("format_version") == 1 and manifest.get("kind") == "policy_export", (
        "invalid policy manifest"
    )
    hashes = manifest.get("sha256")
    assert isinstance(hashes, dict), "missing checkpoint hashes"
    for name in (config_name, "model.safetensors"):
        assert sha256(directory / name) == hashes.get(name), f"checkpoint checksum mismatch: {name}"


def write_policy_export(
    directory, config: PolicyConfig, state_dict: dict[str, Tensor], *, weight_profile: str
):
    """Write the inference export (``config.json``, ``model.safetensors``, ``manifest.json``).

    ``state_dict`` carries the policy's ``dit.`` keys; ``weight_profile`` names their origin
    (``model`` or an EMA profile) in the manifest. The recorded dtype is the tensors' dtype.
    """
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("config.json", "model.safetensors", "manifest.json")):
        raise FileExistsError("export destination already contains a policy")
    dtypes = {v.dtype for v in state_dict.values() if v.is_floating_point()}
    assert len(dtypes) == 1, "export weights need one dtype"
    dtype = str(dtypes.pop()).removeprefix("torch.")
    config = replace(config, trunk_weights=None, torch_dtype=dtype).to_dict()
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in state_dict.items()},
        str(directory / "model.safetensors"),
    )
    manifest = {
        "format_version": 1,
        "kind": "policy_export",
        "weight_profile": weight_profile,
        "sha256": {name: sha256(directory / name) for name in ("config.json", "model.safetensors")},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


@dataclass
class FrozenComponents:
    """Video VAE and text encoder: eval-only, never saved, never trained.

    Deliberately not an ``nn.Module``: assigning an instance to the policy registers nothing, so both
    components stay out of ``state_dict()`` / ``parameters()`` (never written to ``model.safetensors``,
    never handed to the optimizer) and ``policy.train()`` cannot undo the ``eval()`` applied here. The
    policy's ``_apply`` forwards dtype changes and device moves; an offloaded encoder stays on CPU.
    """

    video_vae: Any
    text_encoder: nn.Module

    def __post_init__(self) -> None:
        for module in self.modules():
            module.eval()
            module.requires_grad_(False)

    def modules(self) -> Iterator[nn.Module]:
        """The torch modules held here (the VAE may be a thin wrapper around its ``module``)."""
        for m in (getattr(self.video_vae, "module", self.video_vae), self.text_encoder):
            if isinstance(m, nn.Module):
                yield m


WINDOW_SEED = "window_seed"


@dataclass
class PreparedWindows:
    """Encoder outputs of a training micro-batch, ready for the DiT (see :meth:`FluxActionPolicy.prepare`).

    ``event`` is set by a caller that ran :meth:`FluxActionPolicy.prepare` on another CUDA stream; the
    DiT forward waits for it before touching the tensors.
    """

    idx: list[int]  # windows of the batch that are trained on
    latents: Tensor  # (n, 96, latent frames, h, w)
    ctxs: list[Tensor]  # per window (1, L_i, ctx_dim) in the DiT dtype
    state: Tensor  # (n, D), gripper-flipped and normalized
    actions: Tensor  # (n, chunk, D), gripper-flipped targets (absolute or deltas), normalized
    action_mask: Tensor | None = None  # (n, D) 1/0 per action dim; None = all dims trained (see flow_loss)
    event: Any = None

    def tensors(self):
        yield self.latents
        yield from self.ctxs
        yield self.state
        yield self.actions
        if self.action_mask is not None:
            yield self.action_mask


class FluxActionPolicy(nn.Module):
    def __init__(
        self,
        config: PolicyConfig,
        *,
        video_vae=None,
        text_encoder=None,
        _restore=False,
    ):
        super().__init__()
        assert video_vae is not None or config.video_vae_id, "missing video VAE"
        assert text_encoder is not None or config.text_encoder_id, "missing text encoder"
        self.config = config
        self.modality = config.action_modality
        self.dtype_ = getattr(torch, config.torch_dtype)
        # Set by the trainer when parameters are float32 masters computed in bfloat16 (FSDP mixed precision).
        self.compute_dtype: torch.dtype | None = None
        trunk = (
            resolve_weights(config.trunk_weights, TRUNK_WEIGHTS_FILENAME) if config.trunk_weights else None
        )
        if config.content_streams is None:
            config.content_streams = (
                REQUIRED_CONTENT_STREAMS
                if config.predict_future_video or config.video_prune_after_block is not None
                else ACTION_ONLY_CONTENT_STREAMS
            )
        self.dit_params = action_dit_params(
            restrict_content_streams(JointSingleSeqParams(**config.dit_config), config.content_streams),
            self.modality,
            config.action_dim,
            attn_mode=config.attn_mode,
            conditioning_channels=config.conditioning_channels,
        )
        self.dit = build_action_dit(
            self.dit_params,
            None if _restore else trunk,
            modality=self.modality,
            device="meta" if _restore else "cpu",
            dtype=self.dtype_,
            head_seed=config.head_init_seed,
        )
        if video_vae is None:
            from .models.video_vae import load_video_vae

            video_vae = load_video_vae(config.video_vae_id, compile_model=config.compile_model)
        if text_encoder is None:
            from .models.text_encoder import load_text_encoder

            text_encoder = load_text_encoder(config.text_encoder_id, compile_model=config.compile_model)
        self.frozen = FrozenComponents(video_vae, text_encoder)
        self._ctx_cache = {}
        self._offload_text_encoder = False
        self.serving_setup = None
        self._inference_prepared = False
        self._parallel_cfg_branch: str | None = None
        self._parallel_cfg_group = None
        self._parallel_cfg_tp_size = 1
        self.reset()

    def save_pretrained(self, directory):
        """Export DiT/config only. This is not a resumable training checkpoint."""
        assert isinstance(self.dit, JointSingleSeq), "cannot export prepared inference weights"
        write_policy_export(directory, self.config, self.state_dict(), weight_profile="model")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        video_vae=None,
        text_encoder=None,
        device="cpu",
        revision: str | None = None,
        subfolder: str | None = None,
        predict_future_video: bool | None = None,
        config_overrides: dict[str, Any] | None = None,
    ):
        """Load a local or Hugging Face policy package for inference.

        Released DROID packages load BF16 or native-FP8r weights. Call
        :meth:`prepare_inference` after placement to install the serving backend.
        """
        from safetensors import safe_open
        from safetensors.torch import load_file

        directory = resolve_policy_directory(
            pretrained_model_name_or_path, revision=revision, subfolder=subfolder
        )
        config_name = "config.native.json" if (directory / "config.native.json").is_file() else "config.json"
        _verify_policy_manifest(directory, config_name)
        saved_config = PolicyConfig(**json.loads((directory / config_name).read_text()))
        raw_config = saved_config.to_dict()
        raw_config.update(config_overrides or {})
        if predict_future_video is not None:
            raw_config["predict_future_video"] = bool(predict_future_video)
            if not predict_future_video and raw_config.get("dit_config", {}).get(
                "video_prune_after_block"
            ) is None:
                raw_config["content_streams"] = ACTION_ONLY_CONTENT_STREAMS
        config = PolicyConfig(**raw_config)
        architecture_override = tuple(config.content_streams or ()) != tuple(
            saved_config.content_streams or ()
        )
        if architecture_override:
            saved_streams = set(saved_config.content_streams or REQUIRED_CONTENT_STREAMS)
            wanted_streams = set(config.content_streams or REQUIRED_CONTENT_STREAMS)
            if not wanted_streams < saved_streams:
                raise ValueError("checkpoint stream overrides may only remove content streams")
            if saved_config.quantization is not None:
                raise ValueError("initialize an action-only student from a BF16 checkpoint, not FP8r")
        if config.quantization != saved_config.quantization:
            raise ValueError("quantization cannot be changed while loading a checkpoint")
        target = torch.device(device)
        model_path = directory / "model.safetensors"

        policy = cls(config, video_vae=video_vae, text_encoder=text_encoder, _restore=True)
        if architecture_override:
            # Read only tensors present in the smaller student. A DROID checkpoint is ~13 GB; loading the
            # removed future-video branch merely to discard it doubles avoidable host/device traffic.
            wanted = policy.state_dict()
            with safe_open(str(model_path), framework="pt", device=str(target)) as checkpoint:
                available = set(checkpoint.keys())
                missing = sorted(set(wanted) - available)
                mismatched = sorted(
                    key
                    for key in wanted.keys() & available
                    if tuple(checkpoint.get_slice(key).get_shape()) != tuple(wanted[key].shape)
                )
                if missing or mismatched:
                    raise RuntimeError(
                        "checkpoint cannot initialize the action-only student: "
                        f"missing={missing[:4]}, shape_mismatch={mismatched[:4]}"
                    )
                state = {key: checkpoint.get_tensor(key).to(wanted[key].dtype) for key in wanted}
        else:
            state = load_file(str(model_path), device=str(target))
        if saved_config.quantization == "fp8r":
            state = {name.removeprefix("dit."): value for name, value in state.items()}
            from .models.transformer_inf_fp8r import FP8RInferenceDiT

            policy.dit = FP8RInferenceDiT.from_quantized_state_dict(state)
        else:
            if not architecture_override and config.torch_dtype != saved_config.torch_dtype:
                wanted = policy.state_dict()
                state = {key: value.to(wanted[key].dtype) for key, value in state.items()}
            policy.load_state_dict(state, strict=True, assign=True)
        return policy.to(target).eval()

    def prepare_inference(self, *, compile: bool = True, tensor_parallel_group=None) -> bool:
        """Install the prepared DROID backend and compile the DiT and VAE encoder.

        BF16 policies stay trainable and saveable until this explicit operation.
        Native FP8r policies already own their prepared weights and only need
        validation and compilation here.
        """
        # Older packages explicitly disable this; serving encodes only the observed frame.
        self.config.single_frame_encode = True
        if isinstance(self.dit, JointSingleSeq):
            if not _released_inference_config(self.config) or self.device.type != "cuda":
                return False
            from .models.transformer_inf_bf16 import BF16InferenceDiT

            self.dit = BF16InferenceDiT.from_state_dict(self.dit.state_dict())
            self.reset()
        else:
            assert _released_inference_config(self.config), (
                "prepared backend requires released DROID geometry"
            )
        if tensor_parallel_group is not None:
            self.dit.enable_tensor_parallel(tensor_parallel_group)
        self.dit.validate_runtime_device()
        if compile and self.dit._compiled_hot is None:
            self.dit.compile_static()
            self.dit.compile_hot()
        if compile:
            self.video_vae.module.compile_encoder()
        self._inference_prepared = True
        return True

    def enable_parallel_cfg(self, branch: str, group, *, tensor_parallel_size: int) -> None:
        """Run one CFG branch per TP replica and reduce the guided prediction across replicas."""
        if branch not in ("positive", "negative"):
            raise ValueError(f"invalid CFG branch {branch!r}")
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        self._parallel_cfg_branch = branch
        self._parallel_cfg_group = group
        self._parallel_cfg_tp_size = tensor_parallel_size

    @property
    def video_vae(self):
        return self.frozen.video_vae

    @property
    def text_encoder(self) -> nn.Module:
        return self.frozen.text_encoder

    def _apply(self, fn, *args, **kwargs):
        assert not self._inference_prepared, "move policy before prepare_inference"
        super()._apply(fn, *args, **kwargs)
        frozen = self.__dict__.get("frozen")
        if frozen is not None:

            def cast_on_cpu(tensor):
                # Discover the requested dtype without uploading the encoder's weights. Keep the rank
                # so memory-format conversions also accept the probe (e.g. channels_last buffers).
                probe = tensor.new_empty((0,) * tensor.ndim)
                return tensor.to(device="cpu", dtype=fn(probe).dtype)

            for module in frozen.modules():
                offloaded = module is frozen.text_encoder and self._offload_text_encoder
                module._apply(cast_on_cpu if offloaded else fn)
        self.reset()
        self.dtype_ = self.compute_dtype or next(self.dit.parameters()).dtype
        return self

    def set_compute_dtype(self, dtype: torch.dtype | None) -> None:
        """Dtype of streams and context fed to the DiT when it differs from the stored parameters."""
        self.compute_dtype = dtype
        self.dtype_ = dtype or next(self.dit.parameters()).dtype
        self._ctx_cache.clear()

    def set_text_encoder_offload(self, enabled: bool) -> None:
        """Keep the text encoder on the CPU; it visits the policy device only to encode a caption that is
        not cached yet (once per instruction with a fixed instruction set). Saves its 8.9 GB of GPU memory."""
        self.text_encoder.to("cpu" if enabled else self.device)
        self._offload_text_encoder = bool(enabled)

    @property
    def device(self) -> torch.device:
        return next(self.dit.parameters()).device

    def freeze_streams(self, streams: tuple[str, ...]) -> list[str]:
        """Exclude whole content streams from training (``requires_grad=False``); returns their parameter names.

        The image and audio streams of the pretrained trunk receive no loss in action finetuning and
        are not shipped, so they are frozen rather than decayed through zero gradients. Frozen
        parameters stay out of ``get_optim_params()``, the power EMAs and the optimizer checkpoint;
        exports take them unchanged from the trunk.
        """
        keys = {f".{s}." for s in streams}
        names = []
        for name, param in self.named_parameters():
            if any(k in name for k in keys):
                param.requires_grad_(False)
                names.append(name)
        return names

    def freeze_conditioning_heads(self) -> list[str]:
        """Freeze final heads whose predictions are discarded by this policy.

        Conditioning streams are always inputs only. A late-pruned policy also keeps the video
        input/mode blocks but deletes video tokens before the output head, so that head is frozen too.
        """
        names = []
        for mode, head in self.dit.final_layer.items():
            unused_video_head = mode == "video" and not self.config.predict_future_video
            if mode.endswith("_cond") or unused_video_head:
                for name, param in head.named_parameters():
                    param.requires_grad_(False)
                    names.append(f"dit.final_layer.{mode}.{name}")
        return names

    def get_optim_params(self) -> list[dict[str, Any]]:
        """Two groups: the trunk and the fresh embodiment heads at their own rate."""
        head_prefixes = tuple(f"dit.{n}." for n in fresh_module_names(self.modality))
        trunk, heads = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (heads if name.startswith(head_prefixes) else trunk).append(p)
        groups: list[dict[str, Any]] = [{"params": trunk, "lr": self.config.optimizer_lr}]
        if heads:
            groups.append(
                {"params": heads, "lr": self.config.optimizer_lr * self.config.optimizer_lr_heads_multiplier}
            )
        return groups

    def reset(self) -> None:
        self._ctx_cache.clear()
        self._prepared_text_cache: dict[str, Any] = {}
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)
        self._last_command: Tensor | None = None
        self._observation_history = history.ObservationHistory(self.config.n_obs_steps)

    # ------------------------------------------------------------------ batch helpers
    def _flip(self, x: Tensor) -> Tensor:
        """``x -> 1 - x`` on the configured gripper dims (last axis). Self-inverse."""
        dims = list(self.config.gripper_flip_dims)
        if not dims:
            return x
        x = x.clone()
        x[..., dims] = 1.0 - x[..., dims]
        return x

    def _cameras(self, batch: dict[str, Any]) -> Tensor:
        """Stack the configured cameras: ``(B, n_cams, T, 3, H, W)`` (``T = 1`` for single frames).

        uint8 frames stay uint8 here; ``materialize_video`` scales them on the compute device, so a
        training batch moves as bytes.
        """
        layout = self.config.camera_layout
        cams = []
        for key in self.config.camera_order:
            if key not in batch:
                have = sorted(k for k in batch if isinstance(k, str) and k.startswith("images."))
                order = " [wrist, left exterior, right exterior]" if layout == "droid" else ""
                raise KeyError(
                    f"batch lacks camera {key!r}; image keys present: {have}. camera_keys must name every "
                    f"stream as it appears in the batch (images.<camera>) in layout order{order}"
                )
            img = batch[key]
            if img.ndim == 4:
                img = img[:, None]
            assert img.ndim == 5, "invalid camera rank"
            assert (img.is_floating_point() or img.dtype == torch.uint8) and img.shape[2] == 3, (
                "invalid camera tensor"
            )
            assert layout != "droid" or tuple(img.shape[-2:]) == (360, 640), "invalid DROID frame"
            cams.append(img)
        if len({tuple(c.shape) for c in cams}) != 1:
            same_btc = len({tuple(c.shape[:3]) for c in cams}) == 1
            assert layout == "grid" and same_btc, "incompatible camera shapes"
            # the grid layout takes mixed resolutions: every camera is resized to the largest one first
            hw = (max(c.shape[-2] for c in cams), max(c.shape[-1] for c in cams))
            cams = [self._resize_camera(c, hw) for c in cams]
        return torch.stack(cams, dim=1)

    @staticmethod
    def _resize_camera(img: Tensor, hw: tuple[int, int]) -> Tensor:
        """``(B, T, C, H, W)`` -> ``(B, T, C, *hw)``, keeping a uint8 input uint8."""
        out = F.interpolate(
            img.flatten(0, 1).float(), size=hw, mode="bilinear", align_corners=False, antialias=True
        ).unflatten(0, img.shape[:2])
        if img.dtype == torch.uint8:
            out = out.round_().clamp_(0, 255).to(torch.uint8)
        return out

    def _state(self, batch: dict[str, Any]) -> Tensor:
        state = batch[OBS_STATE]
        if state.ndim == 3:
            assert state.shape[1] == 1, "expected one state history frame"
            state = state[:, 0]  # a length-1 history from the dataset window
        assert state.ndim == 2 and state.shape[1] == self.config.action_dim, "invalid state shape"
        return state.float()

    # ------------------------------------------------------------------ action space
    def _normalize_state(self, state: Tensor) -> Tensor:
        """Flipped state in dataset units ``(B, D)`` -> the state token's values."""
        return normalization.normalize(state, self.config.state_normalization, self.config.normalization_clip)

    def _action_targets(self, actions: Tensor, previous: Tensor | None) -> Tensor:
        """Flipped commands ``(B, K, D)`` -> normalized training targets: the commands themselves, or their
        per-frame deltas from ``previous`` (the command before the window) under ``joint_delta``."""
        cfg = self.config
        if cfg.action_parameterization == "joint_delta":
            assert previous is not None, "joint deltas require action_prev"
            actions = normalization.deltas(actions, self._flip(previous.float()), cfg.absolute_action_dims)
        return normalization.normalize(actions, cfg.action_normalization, cfg.normalization_clip)

    def _targets_to_actions(self, targets: Tensor, state: Tensor) -> Tensor:
        """Predicted normalized targets ``(K, D)`` and the flipped observed ``state (D,)`` in dataset units ->
        flipped absolute commands ``(K, D)`` in dataset units (deltas integrated onto the state)."""
        cfg = self.config
        targets = normalization.denormalize(targets, cfg.action_normalization)
        if cfg.action_parameterization == "joint_delta":
            targets = normalization.integrate(targets, state, cfg.absolute_action_dims)
        return targets

    @staticmethod
    def _captions(batch: dict[str, Any], batch_size: int) -> list[str]:
        task = batch.get("task")
        if task is None:
            return [""] * batch_size
        if isinstance(task, str):
            return [task] * batch_size
        task = list(task)
        if len(task) == 1 and batch_size > 1:
            task = task * batch_size
        assert len(task) == batch_size, "caption batch mismatch"
        return ["" if t is None else str(t) for t in task]

    @staticmethod
    def _valid_windows(batch: dict[str, Any], batch_size: int, device: torch.device) -> Tensor:
        """Windows that reach past the episode end carry ``*_is_pad`` flags; those samples are excluded."""
        keep = torch.ones(batch_size, dtype=torch.bool, device=device)
        for key, value in batch.items():
            if (
                isinstance(key, str)
                and key.endswith("_is_pad")
                and isinstance(value, Tensor)
                and value.dtype == torch.bool
                and value.ndim >= 1
                and value.shape[0] == batch_size
            ):
                keep &= ~value.reshape(batch_size, -1).any(-1).to(device)
        return keep

    def _context(self, caption: str, device: torch.device) -> tuple[Tensor, Tensor]:
        """Text context ``(1, L, ctx_dim)`` in the DiT dtype plus its position ids, cached per caption."""
        fixed_length = self.config.text_fixed_length if self.config.inference_profile == "history" else None
        return self._contexts([caption], device, fixed_length=fixed_length)[0]

    def _contexts(
        self, captions: list[str], device: torch.device, *, fixed_length: int | None = None
    ) -> list[tuple[Tensor, Tensor]]:
        """Contexts of many captions; the ones not cached are encoded together (one call per length bucket)."""
        keys = {c: c if fixed_length is None else (c, fixed_length) for c in captions}
        wanted = set(captions)
        todo = sorted(
            c for c in wanted if (hit := self._ctx_cache.get(keys[c])) is None or hit[0].device != device
        )
        if todo:
            if len(self._ctx_cache) + len(todo) > 256:
                # Bounded cache: start over, and encode every caption of this batch so that the ones that were
                # hits a moment ago are present again.
                self._ctx_cache.clear()
                self._prepared_text_cache.clear()
                todo = sorted(wanted)
            encoder = self.text_encoder
            try:
                if self._offload_text_encoder:
                    encoder.to(device)
                encoded = text_contexts(encoder, todo, device, fixed_length=fixed_length)
            finally:
                if self._offload_text_encoder:
                    encoder.to("cpu")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            for caption, ctx in zip(todo, encoded, strict=True):
                self._ctx_cache[keys[caption]] = (
                    ctx.to(self.dtype_),
                    packing.pack_text(ctx, VEC_DIM)["ctx_ids"],
                )
        return [self._ctx_cache[keys[c]] for c in captions]

    def _cast_inputs(self, kwargs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Streams, context and vector in the DiT dtype; timesteps stay fp32 (bf16 collapses t near 1); ids stay int."""
        out = {}
        for k, v in kwargs.items():
            if v.is_floating_point() and not (k.endswith("_timesteps") or k == "timesteps_ctx"):
                v = v.to(self.dtype_)
            out[k] = v
        return out

    # ------------------------------------------------------------------ training
    @staticmethod
    def _window_generators(batch: dict[str, Any], batch_size: int) -> list[torch.Generator | None]:
        """One CPU generator per window seeded from the loader's ``window_seed``, or ``None`` (global RNG).

        With seeds, a window's caption dropout and augmentation depend on the window alone: not on the
        rank that trains it, nor on when it is encoded (``prepare`` may run ahead of the DiT step), so a
        resumed run reproduces them exactly.
        """
        seeds = batch.get(WINDOW_SEED)
        if seeds is None:
            return [None] * batch_size
        seeds = seeds.tolist() if isinstance(seeds, Tensor) else list(seeds)
        assert len(seeds) == batch_size, "window seed batch mismatch"
        return [torch.Generator().manual_seed(int(s)) for s in seeds]

    @torch.no_grad()
    def prepare(self, batch: dict[str, Any]) -> PreparedWindows:
        """Run everything before the DiT for a training micro-batch: validation, caption dropout,
        augmentation, canvas composition, video-VAE and text encoding.

        Frozen work without gradients; a trainer may run it on a side CUDA stream for the next
        micro-batch while the current one trains (see ``training.trainer``). Randomness comes from the
        per-window generators (``window_seed`` in the batch) when the loader provides them, else from the
        global RNG.
        """
        cfg = self.config
        cams = self._cameras(batch)  # (B, n_cams, T, 3, H, W)
        b, _, t = cams.shape[:3]
        if t != cfg.window_frames:
            raise ValueError(
                f"training needs {cfg.window_frames} frames per camera (observations plus future frames), got {t}"
            )
        device = self.device  # frames may still be on the host; they move per VAE batch below
        if cfg.inference_profile == "history":
            commands = batch.get("command_history")
            if commands is None:
                raise ValueError("history training requires absolute command_history")
            state = history.conditioning_values(batch["state"], commands, cfg)
            previous = commands[:, -1]
        else:
            state = self._normalize_state(self._flip(self._state(batch)))
            previous = batch.get(ACTION_PREV)
        if state.shape[0] != b:
            raise ValueError("state and cameras must have the same batch size")
        state = state.to(device, non_blocking=True)
        actions = self._flip(batch[ACTION].float())  # (B, chunk, D) commands in dataset units
        if actions.shape != (b, cfg.chunk_size, cfg.action_dim):
            raise ValueError(
                f"actions must be (B, {cfg.chunk_size}, {cfg.action_dim}), got {tuple(actions.shape)}"
            )
        actions = self._action_targets(actions, previous).to(device, non_blocking=True)
        keep = self._valid_windows(batch, b, device)
        captions = self._captions(batch, b)
        generators = self._window_generators(batch, b)
        if self.training and cfg.caption_dropout > 0:
            captions = [
                "" if float(torch.rand((), generator=g)) < cfg.caption_dropout else c
                for c, g in zip(captions, generators, strict=True)
            ]
        dropped: list[list[int]] = [[] for _ in range(b)]
        if self.training and cfg.camera_dropout:
            # one draw per configured camera and window, in camera order, after the caption draw
            for i, g in enumerate(generators):
                for c, key in enumerate(cfg.camera_order):
                    p = cfg.camera_dropout.get(key, 0.0)
                    if p > 0 and float(torch.rand((), generator=g)) < p:
                        dropped[i].append(c)
        idx_keep = [i for i, valid in enumerate(keep.tolist()) if valid]
        if not idx_keep:
            empty = torch.zeros(0, device=device)
            return PreparedWindows(idx_keep, empty, [], empty, empty)
        ctxs = [
            ctx
            for ctx, _ in self._contexts(
                [captions[i] for i in idx_keep],
                device,
                fixed_length=cfg.text_fixed_length if cfg.inference_profile == "history" else None,
            )
        ]
        latents = self._encode_windows(cams, idx_keep, device, generators, dropped)  # (n, 96, frames, h, w)
        # Optional per-window action mask (B, D): a dataset that pools embodiments of different action widths
        # into one head pads the narrow ones and masks the padding out of the loss.
        mask = batch.get("action_mask")
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.float32)[idx_keep].to(device, non_blocking=True)
        return PreparedWindows(idx_keep, latents, ctxs, state[idx_keep], actions[idx_keep], mask)

    def forward(
        self,
        batch: dict[str, Any],
        prepared: PreparedWindows | None = None,
        *,
        teacher_dit: nn.Module | None = None,
        teacher_distill_weight: float = 0.0,
        teacher_distill_timestep: float = 0.999,
        teacher_distill_sigma: float = 0.9997998476028442,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Training loss of a micro-batch; ``prepared`` is the output of :meth:`prepare` for this batch
        (computed here when not given)."""
        cfg = self.config
        if prepared is None:
            prepared = self.prepare(batch)
        elif prepared.event is not None:
            # encoded on another stream: order this stream after it and keep the tensors' memory from
            # being recycled while it is still in use here
            stream = torch.cuda.current_stream()
            stream.wait_event(prepared.event)
            for tensor in prepared.tensors():
                tensor.record_stream(stream)
        n = len(prepared.idx)
        if n == 0:
            zero = next(p for p in self.parameters() if p.requires_grad).sum() * 0.0
            return zero, {
                "video_mse": 0.0,
                "action_mse": 0.0,
                "teacher_action_mse": 0.0,
                "teacher_endpoint_action_mae": 0.0,
                "n_valid_windows": 0,
            }
        if cfg.inference_profile == "history":
            video = history.pack_video(prepared.latents, cfg)
            action = history.pack_actions(prepared.state, prepared.actions, cfg)
        else:
            video = packing.pack_video(prepared.latents, fps=cfg.fps, include_future=cfg.predict_future_video)
            action = packing.pack_actions(
                prepared.state[:, None],
                prepared.actions,
                packing.default_action_times(n, cfg.chunk_size, cfg.fps),
                self.modality,
                scale=cfg.action_scale,
            )
        timesteps = packing.sample_timesteps(n, None, cfg.train_timestep_width, cfg.train_timestep_shift)
        action_timesteps = timesteps
        if cfg.separate_timesteps:
            timesteps = torch.sigmoid(torch.randn(n) * cfg.video_logit_std + cfg.video_logit_mean)
        # A late-pruned student receives pure future-video noise as context but never predicts it.
        # Draw it before action noise to preserve the released WAM's random-stream order. Unlike the
        # old action-only student, this makes the teacher target identifiable: teacher and student see
        # the same stochastic variable until the configured joint-block boundary.
        video_context = None
        if cfg.video_prune_after_block is not None:
            n_pred = packing.latent_frames(cfg.window_frames) - 1
            future_zeros = prepared.latents.new_zeros(
                n, packing.LATENT_CHANNELS, n_pred, *cfg.latent_hw
            )
            packed_context = packing.pack_video(
                torch.cat((prepared.latents[:, :, :1], future_zeros), dim=2),
                fps=cfg.fps,
                include_future=True,
            )
            video_noise, _ = packing.add_noise(
                packed_context["x_video"], torch.ones(n, device=packed_context["x_video"].device)
            )
            video_context = (video_noise, packed_context["x_video_ids"])

        kwargs, targets, _ = packing.build_forward_kwargs(
            video,
            action,
            {},
            timesteps,
            self.modality,
            action_timesteps=action_timesteps,
            conditioning_noise_max=cfg.conditioning_noise_max if self.training else 0.0,
        )
        if video_context is not None:
            video_noise, video_ids = video_context
            kwargs.update(
                x_video=video_noise,
                x_video_ids=video_ids.to(video_noise.device),
                x_video_timesteps=torch.ones(video_noise.shape[:2], device=video_noise.device),
            )
        # All windows of the micro-batch go through ONE DiT forward: they are packed into a single
        # sequence whose attention stays within each window (``seqlens``), so captions of different
        # lengths need no padding, the matmuls run over the whole micro-batch, and every rank issues one
        # set of FSDP collectives per micro-step whatever its windows contain.
        packed, seqlens = packing.pack_windows(kwargs, prepared.ctxs, VEC_DIM)
        pred = self.dit(**self._cast_inputs(packed), seqlens=seqlens)
        losses = packing.flow_loss(
            pred,
            packing.flatten_windows(targets),
            self.modality,
            cfg.action_loss_weight,
            cfg.video_loss_weight,
            action_mask=prepared.action_mask,
            reduction=cfg.loss_reduction,
            channel_weights=cfg.action_channel_weights,
        )
        teacher_action_mse = losses["action_mse"].new_zeros(())
        teacher_endpoint_action_mae = teacher_action_mse
        if teacher_dit is not None:
            if cfg.predict_future_video:
                raise ValueError("teacher distillation is only defined for an action-only student")
            if teacher_distill_weight <= 0:
                raise ValueError("teacher_distill_weight must be positive when teacher_dit is provided")
            if not 0 < teacher_distill_timestep <= 1:
                raise ValueError("teacher_distill_timestep must be in (0, 1]")
            if not 0 < teacher_distill_sigma <= 1:
                raise ValueError("teacher_distill_sigma must be in (0, 1]")

            # Match the exact one-step deployment distribution instead of asking the teacher to look at
            # ground-truth future frames.  The full WAM gets eight Gaussian future-video latent frames;
            # teacher and student share the same action noise, observed frame, state and caption. Sampling
            # starts from pure noise while the model sees scheduler tick 999 / 1000, so keep those separate.
            n_pred = packing.latent_frames(cfg.window_frames) - 1
            future_zeros = prepared.latents.new_zeros(
                n, packing.LATENT_CHANNELS, n_pred, *cfg.latent_hw
            )
            teacher_video = packing.pack_video(
                torch.cat((prepared.latents, future_zeros), dim=2),
                fps=cfg.fps,
                include_future=True,
            )
            noise_t = torch.ones(n)
            teacher_kwargs, _, _ = packing.build_forward_kwargs(
                teacher_video,
                action,
                {},
                noise_t,
                self.modality,
                action_timesteps=noise_t,
            )
            teacher_kwargs["x_video_timesteps"].fill_(teacher_distill_timestep)
            teacher_kwargs[f"x_{self.modality}_timesteps"].fill_(teacher_distill_timestep)
            student_kwargs = (
                teacher_kwargs
                if cfg.video_prune_after_block is not None
                else {
                    key: value
                    for key, value in teacher_kwargs.items()
                    if key not in ("x_video", "x_video_ids", "x_video_timesteps")
                }
            )
            teacher_packed, teacher_seqlens = packing.pack_windows(
                teacher_kwargs, prepared.ctxs, VEC_DIM
            )
            student_packed, student_seqlens = packing.pack_windows(
                student_kwargs, prepared.ctxs, VEC_DIM
            )
            student_distill = self.dit(
                **self._cast_inputs(student_packed), seqlens=student_seqlens
            )
            with torch.no_grad():
                teacher_pred = teacher_dit(
                    **self._cast_inputs(teacher_packed), seqlens=teacher_seqlens
                )
            teacher_losses = packing.flow_loss(
                student_distill,
                {f"x_{self.modality}": teacher_pred[f"x_{self.modality}"].detach()},
                self.modality,
                action_weight=1.0,
                action_mask=prepared.action_mask,
                reduction=cfg.loss_reduction,
                channel_weights=cfg.action_channel_weights,
            )
            teacher_action_mse = teacher_losses["action_mse"]
            action_key = f"x_{self.modality}"
            teacher_endpoint_action_mae = (
                (student_distill[action_key].float() - teacher_pred[action_key].float())
                .abs()
                .mean()
                .detach()
                * teacher_distill_sigma
                / cfg.action_scale
            )
            losses["loss"] = losses["loss"] + teacher_distill_weight * teacher_losses["loss"]
        # the MSEs stay on the device (detached): a float() here would wait for the forward and delay
        # the launch of the next micro-batch's encoders
        return losses["loss"], {
            "video_mse": losses["video_mse"],
            "action_mse": losses["action_mse"],
            "teacher_action_mse": teacher_action_mse,
            "teacher_endpoint_action_mae": teacher_endpoint_action_mae,
            "n_valid_windows": n,
        }

    @staticmethod
    def _drop_cameras(cams: Tensor, dropped: list[int]) -> Tensor:
        """``(n_cams, T, 3, H, W)`` with the cameras ``dropped`` replaced by the flat mid-gray tile."""
        if not dropped:
            return cams
        cams = cams.clone()
        cams[dropped] = GRAY_LEVEL if cams.dtype == torch.uint8 else GRAY_LEVEL / 255.0
        return cams

    def _encode_windows(
        self,
        cams: Tensor,
        idx: list[int],
        device: torch.device,
        generators: list[torch.Generator | None],
        dropped: list[list[int]] | None = None,
    ) -> Tensor:
        """Augment, compose and VAE-encode the windows ``idx`` of ``cams``: ``(len(idx), 96, frames, h, w)``.

        The augmentation of window ``i`` is drawn from ``generators[i]`` (the global RNG when ``None``), in
        window order; ``dropped[i]`` lists the cameras of window ``i`` that become gray tiles;
        ``vae_batch_windows`` windows share one VAE call.
        """
        cfg = self.config
        augment = self.training and cfg.augment
        camera_hw = tuple(cams.shape[-2:])
        latents = []
        for start in range(0, len(idx), cfg.vae_batch_windows):
            videos = torch.stack(
                [
                    packing.materialize_video(
                        self._drop_cameras(
                            (
                                cams[i].cpu().float() / 255
                                if cfg.inference_profile == "history" and cams.dtype == torch.uint8
                                else cams[i]
                            )[
                                :,
                                : cfg.window_frames
                                if cfg.predict_future_video
                                else 1,
                            ],
                            dropped[i] if dropped else [],
                        ),
                        packing.sample_augmentation(generators[i], camera_hw=camera_hw) if augment else None,
                        device,
                        layout=cfg.camera_layout,
                        canvas_hw=cfg.canvas_hw,
                    )
                    for i in idx[start : start + cfg.vae_batch_windows]
                ]
            )  # (nb, 3, T, Hc, Wc)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if cfg.predict_future_video:
                    encoded = (
                        history.encode_windows(self.video_vae, videos, cfg)
                        if cfg.inference_profile == "history"
                        else packing.encode_videos(self.video_vae, videos, cfg.latent_hw)
                    )
                else:
                    encoded = packing.encode_frames(
                        self.video_vae,
                        videos[:, :, 0],
                        cfg.latent_hw,
                        single_frame=cfg.single_frame_encode,
                    )
                latents.append(encoded)
        return torch.cat(latents)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _encode_observation(self, cams: Tensor, state: Tensor) -> dict[str, Tensor]:
        """``cams (n_cams, 1, 3, H, W)``, flipped ``state (1, D)`` -> the four conditioning tensors."""
        cfg = self.config
        canvas = packing.materialize_video(
            cams, None, state.device, layout=cfg.camera_layout, canvas_hw=cfg.canvas_hw
        )
        return self._encode_canvas(canvas, state)

    @torch.no_grad()
    def _encode_canvas(self, canvas: Tensor, state: Tensor) -> dict[str, Tensor]:
        """Canvas ``(3, 1, Hc, Wc)`` in ``[-1, 1]`` and flipped ``state (1, D)`` -> conditioning tensors."""
        cfg = self.config
        device = state.device
        canvas = canvas.to(device)
        latent = packing.encode_single_frame(
            self.video_vae, canvas[:, 0], cfg.latent_hw, single_frame=cfg.single_frame_encode
        )  # (1, 96, 1, h, w)
        video, video_ids = batched_prc_vid(
            latent.to(torch.bfloat16), packing.video_time_ids(1, 0, 1, fps=cfg.fps)
        )
        values = state[:, None] * cfg.action_scale
        action, action_ids = batched_prc_action(
            values.transpose(1, 2), times_to_ids(torch.zeros(1, 1, device=state.device))
        )
        return {
            "x_video_cond": video,
            "x_video_cond_ids": video_ids,
            f"x_{self.modality}_cond": action.float(),
            f"x_{self.modality}_cond_ids": action_ids,
        }

    @torch.no_grad()
    def _sample(self, cond: dict[str, Tensor], caption: str, seed: int) -> Tensor:
        """Denoise actions, jointly with future video only for a WAM, into ``(chunk, D)`` targets."""
        cfg, m, mdt = self.config, self.modality, self.dtype_
        ak, ck = f"x_{m}", f"x_{m}_cond"
        device = cond["x_video_cond"].device
        rich_history = cfg.inference_profile == "history"
        rng = torch.Generator().manual_seed(seed)
        video_sample = None
        uses_video_context = cfg.video_prune_after_block is not None
        if cfg.predict_future_video or uses_video_context:
            # Preserve the released WAM's seed contract: its video noise precedes action noise.
            n_pred = (
                packing.latent_frames(cfg.chunk_size)
                if rich_history
                else packing.latent_frames(cfg.window_frames) - 1
            )
            video_noise = torch.randn(
                1, packing.LATENT_CHANNELS, n_pred, *cfg.latent_hw, generator=rng
            )
            video_sample = batched_prc_vid(
                video_noise,
                packing.video_time_ids(
                    n_pred,
                    packing.latent_frames(cfg.n_obs_steps) if rich_history else 1,
                    1,
                    fps=cfg.video_position_fps if rich_history else cfg.fps,
                ),
            )
        times = (
            torch.arange(cfg.chunk_size).float()[None] / cfg.fps
            if rich_history
            else packing.default_action_times(1, cfg.chunk_size, cfg.fps)
        )
        action_noise = torch.randn(1, cfg.action_dim, cfg.chunk_size, generator=rng)
        x_action, x_action_ids = batched_prc_action(action_noise, times_to_ids(times))
        # The solver state stays fp32 (scaled joint targets would lose ~0.01 rad per bf16 round trip);
        # inputs are cast at the model boundary.
        flow = {ak: x_action.to(device)}
        fixed = {
            f"{ak}_ids": x_action_ids.to(device),
            "x_video_cond": cond["x_video_cond"].to(device, mdt),
            "x_video_cond_ids": cond["x_video_cond_ids"].to(device),
            "x_video_cond_timesteps": torch.zeros(1, cond["x_video_cond"].shape[1], device=device),
            ck: cond[ck].to(device, mdt),
            f"{ck}_ids": cond[f"{ck}_ids"].to(device),
            f"{ck}_timesteps": torch.zeros(1, cond[ck].shape[1], device=device),
            "vector": torch.zeros(1, VEC_DIM, device=device, dtype=mdt),
        }
        guidance = {
            ak: cfg.guidance_scale if cfg.guidance_scale_action is None else cfg.guidance_scale_action,
        }
        if video_sample is not None:
            x_video, x_video_ids = video_sample
            fixed["x_video_ids"] = x_video_ids.to(device)
            if cfg.predict_future_video:
                flow = {"x_video": x_video.to(device), **flow}
                guidance = {"x_video": cfg.guidance_scale, **guidance}
            else:
                fixed["x_video"] = x_video.to(device, mdt)
        ctx_c = self._context(caption, device)
        ctx_uc = self._context("", device) if any(g != 1.0 for g in guidance.values()) else None

        if self._inference_prepared:
            return (
                self._sample_prepared(
                    flow=flow,
                    fixed=fixed,
                    caption=caption,
                    ctx_c=ctx_c,
                    ctx_uc=ctx_uc,
                    guidance=guidance,
                )[ak][0].float()
                / cfg.action_scale
            )

        assert isinstance(self.dit, JointSingleSeq), "call prepare_inference before prediction"

        def predict(samples: dict[str, Tensor], t) -> dict[str, Tensor]:
            t = float(t) / 1000.0 if isinstance(t, Tensor) and not t.is_floating_point() else float(t)
            timesteps = {
                f"{ak}_timesteps": torch.full((1, cfg.chunk_size), t, device=device),
            }
            if "x_video" in samples:
                timesteps["x_video_timesteps"] = torch.full(
                    (1, samples["x_video"].shape[1]), t, device=device
                )
            elif "x_video" in fixed:
                timesteps["x_video_timesteps"] = torch.full(
                    (1, fixed["x_video"].shape[1]), t, device=device
                )
            model_in = {k: v.to(mdt) for k, v in samples.items()}
            if ctx_uc is None:  # guidance 1.0 on every stream: a single conditional pass
                ctx, ctx_ids = ctx_c
                pred = self.dit(
                    **model_in,
                    **fixed,
                    **timesteps,
                    ctx=ctx,
                    ctx_ids=ctx_ids,
                    timesteps_ctx=torch.zeros(ctx.shape[:2], device=device),
                )
                pred = {k: pred[k] for k in model_in}
            else:
                pred = sampling.cfg_two_pass(self.dit, model_in, fixed, timesteps, ctx_uc, ctx_c, guidance)
            return {k: v.float() for k, v in pred.items()}

        if cfg.sampler == "cosmos_unipc":
            out = sampling.cosmos_unipc_order2(
                flow, predict, n_steps=cfg.num_inference_steps, shift=cfg.sampler_shift
            )
        else:
            out = sampling.euler(flow, predict, n_steps=cfg.num_inference_steps, alpha=cfg.sampler_shift)
        return out[ak][0].float() / cfg.action_scale

    def _prepared_text(self, caption: str, ctx: tuple[Tensor, Tensor], vector: Tensor):
        cached = self._prepared_text_cache.get(caption)
        if cached is None:
            cached = self.dit.prepare_text(ctx[0], ctx[1], vector)
            self._prepared_text_cache[caption] = cached
        return cached

    def _sample_prepared(
        self,
        *,
        flow: dict[str, Tensor],
        fixed: dict[str, Tensor],
        caption: str,
        ctx_c: tuple[Tensor, Tensor],
        ctx_uc: tuple[Tensor, Tensor] | None,
        guidance: dict[str, float],
    ) -> dict[str, Tensor]:
        """Run the existing solver over an optimized prepared DROID DiT."""
        cfg, model = self.config, self.dit
        action_key = f"x_{self.modality}"
        cond_key = f"{action_key}_cond"
        vector = fixed["vector"]

        def prepare(caption_key: str, ctx: tuple[Tensor, Tensor]):
            text = self._prepared_text(caption_key, ctx, vector)
            return model.prepare_observation(
                text,
                video_ids=fixed["x_video_ids"],
                video_cond=fixed["x_video_cond"],
                video_cond_ids=fixed["x_video_cond_ids"],
                action_ids=fixed[f"{action_key}_ids"],
                action_cond=fixed[cond_key],
                action_cond_ids=fixed[f"{cond_key}_ids"],
            )

        parallel_branch = self._parallel_cfg_branch
        if parallel_branch is None:
            request = prepare(caption, ctx_c)
            if ctx_uc is None:
                negative_request = None
            else:
                # The observation/action-conditioning early blocks do not see text. Reuse the
                # conditional request's encoded conditioning and replace only the prepared text
                # fields for the unconditional branch instead of evaluating those blocks twice.
                negative_text = self._prepared_text("", ctx_uc, vector)
                # The BF16 backend exposes its text fields separately, so it can splice the
                # negative text into the conditional prepared observation. The native-FP8r
                # backend groups those values differently; prepare that request normally while
                # still sharing its dynamic early-content blocks in the hot path below.
                if not hasattr(negative_text, "txt_len"):
                    negative_request = prepare("", ctx_uc)
                elif negative_text.txt_len != request.txt_len:
                    negative_request = prepare("", ctx_uc)
                else:
                    negative_request = replace(
                        request,
                        txt=negative_text.txt,
                        rope=torch.cat(
                            (
                                negative_text.rope,
                                request.rope[:, :, request.txt_len :],
                            ),
                            dim=2,
                        ),
                        single_txt=negative_text.single_txt,
                    )
        else:
            assert ctx_uc is not None, "parallel CFG requires an unconditional branch"
            request = prepare(caption, ctx_c) if parallel_branch == "positive" else prepare("", ctx_uc)
            negative_request = None
        batched_request = None
        if negative_request is not None:
            try:
                batched_request = _concat_prepared_batches(negative_request, request)
            except ValueError:
                # Captions can land in different text-length buckets. Preserve the serial CFG path
                # in that case rather than padding or changing the attention problem.
                pass
        _, ticks = sampling.cosmos_unipc_schedule(cfg.num_inference_steps, cfg.sampler_shift)
        times = ticks.to(device=self.device, dtype=torch.float32) / 1000.0
        steps = model.prepare_steps(batched_request or request, times, times)
        phase = 0

        def predict(samples: dict[str, Tensor], _t) -> dict[str, Tensor]:
            nonlocal phase
            inputs = {
                "video": samples["x_video"].to(torch.bfloat16),
                "action": samples[action_key].to(torch.bfloat16),
            }
            if parallel_branch is not None:
                result = model.forward_prepared(request, steps[phase], **inputs).as_dict()
                combined = {}
                for key in flow:
                    coefficient = (
                        guidance[key] if parallel_branch == "positive" else 1.0 - guidance[key]
                    ) / self._parallel_cfg_tp_size
                    value = result[key].float() * coefficient
                    dist.all_reduce(value, group=self._parallel_cfg_group)
                    combined[key] = value
                result = combined
            elif negative_request is None:
                result = model.forward_prepared(request, steps[phase], **inputs).as_dict()
            elif batched_request is not None:
                paired_inputs = {
                    key: torch.cat((value, value), dim=0) for key, value in inputs.items()
                }
                paired = model.forward_prepared(
                    batched_request,
                    steps[phase],
                    **paired_inputs,
                    shared_cfg_inputs=True,
                ).as_dict()
                result = {}
                for key in flow:
                    negative, positive = paired[key].chunk(2, dim=0)
                    result[key] = negative + guidance[key] * (positive - negative)
            else:
                negative = model.forward_prepared(negative_request, steps[phase], **inputs).as_dict()
                positive = model.forward_prepared(request, steps[phase], **inputs).as_dict()
                result = {
                    key: negative[key] + guidance[key] * (positive[key] - negative[key]) for key in flow
                }
            phase += 1
            return {key: value.float() for key, value in result.items()}

        if cfg.num_inference_steps == 4:
            return sampling.cosmos_unipc_order2_fused4(
                flow,
                predict,
                shift=cfg.sampler_shift,
            )
        return sampling.cosmos_unipc_order2(
            flow, predict, n_steps=cfg.num_inference_steps, shift=cfg.sampler_shift
        )

    @torch.no_grad()
    def training_targets(self, batch: dict[str, Any]) -> Tensor:
        """The normalized targets a training window carries, ``(B, chunk_size, action_dim)``: flip, deltas from
        ``action_prev`` under ``joint_delta``, range normalization. Comparable to :meth:`predict_normalized_targets`."""
        cfg = self.config
        actions = self._flip(batch[ACTION].float())
        if actions.ndim != 3 or actions.shape[1:] != (cfg.chunk_size, cfg.action_dim):
            raise ValueError(
                f"actions must be (B, {cfg.chunk_size}, {cfg.action_dim}), got {tuple(actions.shape)}"
            )
        previous = (
            batch["command_history"][:, -1] if cfg.inference_profile == "history" else batch.get(ACTION_PREV)
        )
        return self._action_targets(actions, previous)

    @torch.no_grad()
    def predict_normalized_targets(self, batch: dict[str, Any]) -> Tensor:
        """Observation batch -> ``(B, chunk_size, action_dim)`` predicted targets in the trained action space:
        normalized, deltas under ``joint_delta``, before denormalization and integration."""
        self.config.validate_inference()
        if self._inference_prepared and self.dit._compiled_hot is not None:
            torch.compiler.cudagraph_mark_step_begin()
        if self.training:
            # Prediction enters eval once; validation callers may resume training.
            self.eval()
        if self.config.inference_profile == "history":
            batch = history.observation_window(batch, self.config)
            cams = self._cameras(batch)
            state, commands = batch["state"], batch["command_history"]
            if cams.shape[0] != state.shape[0] or cams.device != state.device:
                raise ValueError("state and cameras must have the same batch size and device")
            return torch.stack(
                [
                    self._sample(
                        history.pack_conditioning(
                            self.video_vae, cams[i], state[i : i + 1], commands[i : i + 1], self.config
                        ),
                        caption,
                        self.config.inference_seed,
                    )
                    for i, caption in enumerate(self._captions(batch, state.shape[0]))
                ]
            )
        cams = self._cameras(batch)[:, :, :1]  # the current frame (index 0 of a window)
        state = self._flip(self._state(batch))  # (B, D) in dataset units
        b = state.shape[0]
        assert cams.shape[0] == b and cams.device == state.device, "observation batch mismatch"
        captions = self._captions(batch, b)
        token = self._normalize_state(state)
        targets = []
        for i in range(b):
            cond = self._encode_observation(cams[i], token[i : i + 1])
            targets.append(self._sample(cond, captions[i], self.config.inference_seed))
        return torch.stack(targets)

    @torch.no_grad()
    def actions_from_targets(self, targets: Tensor, state: Tensor) -> Tensor:
        """Normalized targets ``(B, chunk_size, action_dim)`` and the observed ``state (B, action_dim)`` in dataset
        units -> absolute commands in dataset units (gripper as stored)."""
        flipped = self._flip(state.float())
        chunks = [self._targets_to_actions(t, s.to(t.device)) for t, s in zip(targets, flipped, strict=True)]
        return self._flip(torch.stack(chunks)).float()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """Observation batch -> ``(B, chunk_size, action_dim)`` absolute commands in the dataset's action units.

        Normalization is undone and, under ``joint_delta``, the predicted deltas are integrated onto the
        observed state, so the result is what the dataset's action column holds (gripper as stored).
        """
        if kwargs:
            raise NotImplementedError("FLUX Action does not implement RTC inference arguments yet")
        if self.config.inference_profile == "history":
            batch = history.observation_window(batch, self.config)
            targets = self.predict_normalized_targets(batch)
            return self.actions_from_targets(targets, batch["command_history"][:, -1])
        targets = self.predict_normalized_targets(batch)
        return self.actions_from_targets(targets, self._state(batch))

    @torch.inference_mode()
    def predict_from_composite(
        self, composite: Tensor, state: Tensor, task: str, *, seed: int | None = None
    ) -> Tensor:
        """DROID composite view -> ``(chunk_size, action_dim)`` actions in the dataset's conventions.

        ``composite``: the 540x640 view a RoboLab or Cosmos client sends (wrist on top, the two
        exteriors at half resolution below), uint8 ``(H, W, 3)`` or float ``(3, H, W)`` in ``[0, 1]``.
        ``state``: ``(action_dim,)`` in the dataset's conventions (gripper as stored); the gripper
        flip happens here on the way in and out, as in training. Placement on the canvas equals the
        three-camera path, so a composite built like the training composite gives identical actions.
        """
        cfg = self.config
        assert cfg.camera_layout == "droid", "composite inference requires DROID layout"
        self.config.validate_inference()
        if self._inference_prepared and self.dit._compiled_hot is not None:
            torch.compiler.cudagraph_mark_step_begin()
        if self.training:
            # Prediction enters eval once; validation callers may resume training.
            self.eval()
        if composite.dtype == torch.uint8:
            assert composite.ndim == 3 and composite.shape[-1] == 3, "invalid uint8 composite"
            composite = composite.permute(2, 0, 1).float().div_(255.0)
        else:
            assert composite.ndim == 3 and composite.shape[0] == 3, "invalid float composite"
            assert composite.is_floating_point(), "expected float composite"
        device = self.device
        canvas = packing.pad_composite(composite[None].to(device), cfg.canvas_hw)  # (3, 1, Hc, Wc)
        flipped = self._flip(state.float().reshape(1, -1)).to(device)
        assert flipped.shape[1] == cfg.action_dim, "invalid state width"
        cond = self._encode_canvas(canvas, self._normalize_state(flipped))
        chunk = self._sample(cond, task, cfg.inference_seed if seed is None else seed)
        return self._flip(self._targets_to_actions(chunk, flipped[0].to(chunk.device))).float()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """One absolute command per call from a queue of ``n_action_steps`` predictions (open loop; a
        ``joint_delta`` policy continues from its last returned command, initially the observed state).

        Reset between episodes or interventions. Measured state still conditions every new prediction.
        """
        if kwargs:
            raise NotImplementedError("FLUX Action does not implement RTC inference arguments yet")
        if self.config.inference_profile == "history":
            window = self._observation_history.append(batch, self.config, self._last_command)
            if not self._action_queue:
                targets = self.predict_normalized_targets(window)[:, : self.config.n_action_steps]
                self._action_queue.extend(targets.transpose(0, 1))
            target = self._action_queue.popleft()
            action = self.actions_from_targets(target[:, None], window["command_history"][:, -1])[:, 0]
            self._last_command = action.detach().clone()
            return action
        if len(self._action_queue) == 0:
            if self.config.action_parameterization == "joint_delta":
                targets = self.predict_normalized_targets(batch)
                anchor = self._state(batch) if self._last_command is None else self._last_command
                actions = self.actions_from_targets(targets, anchor)
            else:
                actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions[:, : self.config.n_action_steps].transpose(0, 1))
        action = self._action_queue.popleft()
        if self.config.action_parameterization == "joint_delta":
            self._last_command = action.detach().clone()
        return action


def _concat_prepared_batches(first, second):
    """Concatenate two compatible frozen prepared-state dataclasses along their batch dimension."""

    if type(first) is not type(second):
        raise ValueError("prepared state types differ")

    def merge(left, right):
        if isinstance(left, Tensor):
            if not isinstance(right, Tensor) or left.shape[1:] != right.shape[1:]:
                raise ValueError("prepared tensor shapes differ")
            return torch.cat((left, right), dim=0)
        if isinstance(left, tuple):
            if not isinstance(right, tuple) or len(left) != len(right):
                raise ValueError("prepared tuple shapes differ")
            return tuple(merge(a, b) for a, b in zip(left, right, strict=True))
        if left != right:
            raise ValueError("prepared metadata differs")
        return left

    updates = {}
    for field in fields(first):
        left, right = getattr(first, field.name), getattr(second, field.name)
        if field.name in ("batch", "batch_size"):
            updates[field.name] = left + right
        else:
            updates[field.name] = merge(left, right)
    return replace(first, **updates)
