"""
Probe inference script for LongLive causal memory diagnostics.

Runs interactive (multi-prompt) inference exactly as the original pipeline,
but calls JacobianProbe at selected blocks around prompt switches.

Usage:
    python probe_inference.py \
        --config_path configs/longlive_interactive_inference.yaml \
        [--probe-layers 0,10,20,29] \
        [--num-proj 2] \
        [--probe-seed 1234] \
        [--output-dir outputs/probes/run0] \
        [--run-name run0]

The original inference trajectory is preserved exactly.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange

from utils.misc import set_seed
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
from utils.dataset import MultiTextDataset
from probes.jacobian_probe import JacobianProbe


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser("Jacobian probe inference")
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--probe-layers", type=str, default=None,
                    help="Comma-separated layer indices, e.g. '0,10,20,29'")
parser.add_argument("--num-proj", type=int, default=2,
                    help="Number of random VJP projections (M)")
parser.add_argument("--probe-seed", type=int, default=1234)
parser.add_argument("--output-dir", type=str, default="outputs/probes/run0")
parser.add_argument("--run-name", type=str, default="run0")
args_cli = parser.parse_args()

config = OmegaConf.load(args_cli.config_path)

# ---------------------------------------------------------------------------
# Distributed setup (single-GPU assumed for probing)
# ---------------------------------------------------------------------------
local_rank = 0
rank = 0
device = torch.device("cuda")
set_seed(config.seed)
print(f"Single GPU probe mode on device {device}")

# Keep grads disabled globally; probes re-enable locally
torch.set_grad_enabled(False)

# ---------------------------------------------------------------------------
# Build pipeline (mirrors interactive_inference.py)
# ---------------------------------------------------------------------------
pipeline = InteractiveCausalInferencePipeline(config, device=device)

if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    raw_gen = state_dict.get("generator_ema" if config.use_ema else "generator")
    if raw_gen is None:
        raw_gen = state_dict
    if config.use_ema:
        cleaned = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in raw_gen.items()}
        pipeline.generator.load_state_dict(cleaned, strict=False)
    else:
        pipeline.generator.load_state_dict(raw_gen)

from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=True,
    )
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        lora_ckpt = torch.load(lora_ckpt_path, map_location="cpu")
        if isinstance(lora_ckpt, dict) and "generator_lora" in lora_ckpt:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_ckpt["generator_lora"])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_ckpt)
    pipeline.is_lora_enabled = True

pipeline = pipeline.to(dtype=torch.bfloat16)
low_memory = get_cuda_free_memory_gb(device) < 40
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)
pipeline.generator.model.eval()

# ---------------------------------------------------------------------------
# Build probe
# ---------------------------------------------------------------------------
num_transformer_blocks = pipeline.num_transformer_blocks  # 30

if args_cli.probe_layers is not None:
    probe_layers = [int(x) for x in args_cli.probe_layers.split(",")]
else:
    L = num_transformer_blocks
    probe_layers = sorted({0, L // 3, 2 * L // 3, L - 1})

print(f"Probe layers: {probe_layers}")

output_dir = os.path.join(args_cli.output_dir, args_cli.run_name)
probe = JacobianProbe(
    enabled=True,
    layers=probe_layers,
    num_layers_total=num_transformer_blocks,
    num_proj=args_cli.num_proj,
    probe_seed=args_cli.probe_seed,
    output_dir=output_dir,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
switch_frame_indices: List[int]
if isinstance(config.switch_frame_indices, int):
    switch_frame_indices = [int(config.switch_frame_indices)]
else:
    switch_frame_indices = [int(x) for x in str(config.switch_frame_indices).split(",") if str(x).strip()]

dataset = MultiTextDataset(config.data_path)
num_segments = len(dataset[0]["prompts_list"])
assert len(switch_frame_indices) == num_segments - 1, (
    "switch_frame_indices length mismatch"
)
print(f"Segments: {num_segments}, switch frames: {switch_frame_indices}")

os.makedirs(config.output_folder, exist_ok=True)

# Only run on the first prompt for the diagnostic (override inference_iter=0)
batch_data = dataset[0]
prompts_list: List[List[str]] = batch_data["prompts_list"]

# ---------------------------------------------------------------------------
# Sanity: record a parameter checksum before probing
# ---------------------------------------------------------------------------
_param_ref = {
    n: p.detach().clone()
    for n, p in list(pipeline.generator.model.named_parameters())[:3]
}


def _check_params_unchanged():
    for n, ref in _param_ref.items():
        cur = dict(pipeline.generator.model.named_parameters())[n]
        if not torch.allclose(ref, cur.detach()):
            print(f"[SANITY FAIL] Parameter {n} changed during probing!")
        else:
            print(f"[SANITY OK] Parameter {n} unchanged.")


# ---------------------------------------------------------------------------
# Probe-augmented inference  (mirrors InteractiveCausalInferencePipeline.inference)
# ---------------------------------------------------------------------------

batch_size = config.num_samples
num_output_frames = config.num_output_frames

sampled_noise = torch.randn(
    [batch_size, num_output_frames, 16, 60, 104],
    device=device,
    dtype=torch.bfloat16,
    generator=torch.Generator(device=device).manual_seed(config.seed),
)

# Encode all prompts
cond_list = [pipeline.text_encoder(text_prompts=p) for p in prompts_list]

if low_memory:
    from utils.memory import move_model_to_device_with_memory_preservation
    gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
    move_model_to_device_with_memory_preservation(
        pipeline.text_encoder, target_device=gpu,
        preserved_memory_gb=gpu_memory_preservation,
    )

output_device = torch.device("cpu") if low_memory else device
output = torch.zeros(
    [batch_size, num_output_frames, 16, 60, 104],
    device=output_device,
    dtype=sampled_noise.dtype,
)

local_attn_cfg = getattr(config.model_kwargs, "local_attn_size", -1)
if local_attn_cfg != -1:
    kv_cache_size = local_attn_cfg * pipeline.frame_seq_length
else:
    kv_cache_size = num_output_frames * pipeline.frame_seq_length
print(f"kv_cache_size: {kv_cache_size}")

pipeline._initialize_kv_cache(
    batch_size, dtype=sampled_noise.dtype, device=device,
    kv_cache_size_override=kv_cache_size,
)
pipeline._initialize_crossattn_cache(
    batch_size=batch_size, dtype=sampled_noise.dtype, device=device,
)

current_start_frame = 0
pipeline.generator.model.local_attn_size = pipeline.local_attn_size
pipeline._set_all_modules_max_attention_size(pipeline.local_attn_size)

num_frame_per_block = pipeline.num_frame_per_block
num_blocks = num_output_frames // num_frame_per_block
all_num_frames = [num_frame_per_block] * num_blocks

segment_idx = 0
next_switch_pos = switch_frame_indices[segment_idx] if segment_idx < len(switch_frame_indices) else None

# Determine which chunks to probe:
# For each switch at frame T_switch:
#   probe chunk index = T_switch // num_frame_per_block  → "switch"
#   probe chunk index - 1                                → "pre"
#   probe chunk index + 1                                → "post1"
#   probe chunk index + 2                                → "post2"
# Also probe chunk 0 as "baseline"

probe_chunk_meta: dict = {}  # chunk_idx -> relative_to_switch

# Baseline: first chunk
probe_chunk_meta[0] = ("baseline", -1, 0)  # (rel, switch_frame, segment)

for switch_frame in switch_frame_indices:
    switch_chunk = switch_frame // num_frame_per_block
    for offset, label in [(-1, "pre"), (0, "switch"), (1, "post1"), (2, "post2")]:
        c = switch_chunk + offset
        if 0 <= c < num_blocks:
            # Don't overwrite if already set to a closer switch
            if c not in probe_chunk_meta:
                probe_chunk_meta[c] = (label, switch_frame, switch_chunk)

print(f"Probe chunk schedule: {probe_chunk_meta}")

# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
print("\n=== Starting probe inference ===")

for block_idx, current_num_frames in enumerate(all_num_frames):
    if next_switch_pos is not None and current_start_frame >= next_switch_pos:
        segment_idx += 1
        pipeline._recache_after_switch(output, current_start_frame, cond_list[segment_idx])
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )
    cond_in_use = cond_list[segment_idx]

    noisy_input = sampled_noise[:, current_start_frame:current_start_frame + current_num_frames]

    # Spatial denoising loop
    for step_index, current_timestep in enumerate(pipeline.denoising_step_list):
        timestep = (
            torch.ones([batch_size, current_num_frames], device=device, dtype=torch.int64)
            * current_timestep
        )

        if step_index < len(pipeline.denoising_step_list) - 1:
            _, denoised_pred = pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=cond_in_use,
                timestep=timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )
            next_ts = pipeline.denoising_step_list[step_index + 1]
            noisy_input = pipeline.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                next_ts * torch.ones([batch_size * current_num_frames],
                                      device=device, dtype=torch.long),
            ).unflatten(0, denoised_pred.shape[:2])
        else:
            _, denoised_pred = pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=cond_in_use,
                timestep=timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )

    output[:, current_start_frame:current_start_frame + current_num_frames] = \
        denoised_pred.to(output.device)

    # Cache-update forward (context_timestep)
    context_timestep = torch.ones_like(timestep) * config.context_noise

    pipeline.generator(
        noisy_image_or_video=denoised_pred,
        conditional_dict=cond_in_use,
        timestep=context_timestep,
        kv_cache=pipeline.kv_cache1,
        crossattn_cache=pipeline.crossattn_cache,
        current_start=current_start_frame * pipeline.frame_seq_length,
    )

    # ---- Probing ----
    if block_idx in probe_chunk_meta:
        rel_label, switch_frame_ref, switch_chunk_ref = probe_chunk_meta[block_idx]
        print(f"[Probe] block={block_idx}  rel={rel_label}  seg={segment_idx}  ts={int(timestep[0, 0])}")

        probe.collect_probe_at_block(
            global_chunk=block_idx,
            prompt_segment_id=segment_idx,
            relative_to_switch=rel_label,
            denoising_timestep=int(timestep[0, 0]),

            generator_model=pipeline.generator,

            # For J_c/J_x: inputs to the cache-update (context) forward
            context_latent=denoised_pred,
            conditional_dict=cond_in_use,
            context_timestep=context_timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,

            # For J_o: the final denoising forward inputs
            noisy_input=noisy_input,
            denoising_ts=timestep,
        )

    current_start_frame += current_num_frames

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
_check_params_unchanged()

# ---------------------------------------------------------------------------
# Save probe records
# ---------------------------------------------------------------------------
records_path = probe.finalize()

# ---------------------------------------------------------------------------
# Decode and save video (same as original pipeline)
# ---------------------------------------------------------------------------
print("\nDecoding video...")
with torch.no_grad():
    video = pipeline.vae.decode_to_pixel(output.to(device), use_cache=False)
video = (video * 0.5 + 0.5).clamp(0, 1)
current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0
video_path = os.path.join(config.output_folder, f"probe_{args_cli.run_name}.mp4")
write_video(video_path, current_video[0].to(torch.uint8), fps=16)
print(f"Video saved to: {video_path}")

print(f"\n=== Probe inference complete ===")
print(f"Records: {records_path}")
print(f"To generate plots: python probes/visualize_probe.py {records_path}")
