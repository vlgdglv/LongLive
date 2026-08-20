from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from typing import Any, Dict, List, Optional

import torch

log = logging.getLogger(__name__)


class LatentCollector:

    def __init__(
        self,
        seed: int,
        enabled: bool = False,

        output_dir: str = "outputs/probes",
        latent_fps: float = 4.0,
        sampled_id: str = "sample",
    ):
        self.enabled = enabled
        self.seed = seed
        self.output_dir = os.path.join(output_dir, f"latent_{seed}")
        self.latent_fps = latent_fps
        self.sample_id = sampled_id

        if not enabled:
            return


        # Rolling deque of (block_index, cpu_tensor [B, nfpb, C, H, W])
        # self._block_buffer: deque = deque()
        self._block_buffer_dict = {}
        self._buffer_total_frames_dict = {}

        self.capture_kv_layers = [0, 15, 29]

        # Saved records (written by finalize())
        self._records: List[Dict[str, Any]] = []
        self._logged_shape: bool = False

        # Prepare output directories
        os.makedirs(self.output_dir, exist_ok=True)

        log.info(
            "[Collector] latent collector enabled."
        )
        self.pre_segment_idx = -1


    def capture_latent(
        self,
        denoised_block: torch.Tensor,
        noisy_input: torch.Tensor,
        global_block: int,
        current_denoise_timestep_index: int,
        current_denoise_timestep: float,
        max_denoise_step: int,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompts_id: int = 0,
    ) -> None:
        if not self.enabled:
            return
        
        cpu_noise = noisy_input.detach().to("cpu", non_blocking=False)
        cpu_block = denoised_block.detach().to("cpu", non_blocking=False) # [B, F, C, H, W]
        # self._block_buffer_dict[current_denoise_timestep_index].append((global_block, cpu_block))
        # self._buffer_total_frames_dict[current_denoise_timestep_index] += cpu_block.shape[1]

        self._block_buffer_dict[current_denoise_timestep_index] = {
            "step_idx": current_denoise_timestep_index,
            "input_timestep": current_denoise_timestep,
            "input_before_forward": cpu_noise,
            "pred_x0": cpu_block,
        }

        if not self._logged_shape:
            print(
                "[LatentCollector] latent shape = [B={}, F={}, C={}, H={}, W={}], "
                "actual duration per block = {:.3f} sec".format(
                *cpu_block.shape, cpu_block.shape[1] / self.latent_fps
                )
            )
            self._logged_shape = True

        embed_path = os.path.join(self.output_dir, f"prompt{prompts_id}_embeds.pt")
        if prompt_embeds is not None and self.pre_segment_idx!=prompts_id:
            if not os.path.exists(embed_path):
                torch.save(prompt_embeds.detach().to("cpu"), embed_path)
            print("[LatentCollector] prompt {} saved.".format(prompts_id))
        self.pre_segment_idx = prompts_id

        if current_denoise_timestep_index == max_denoise_step-1:
            block_path = os.path.join(self.output_dir, f"latents_{global_block}_{global_block+cpu_block.shape[1]}.pt")
            for k, v in  self._block_buffer_dict.items():
                print(k, ": ",v["input_timestep"], v["input_before_forward"].shape, v["pred_x0"].shape, )
            torch.save({
                "latents": self._block_buffer_dict[current_denoise_timestep_index],
                "embed_path": embed_path,
            }, block_path)
        
            print(
                "[LatentCollector] Captured block [{}, {}]. saved to {}".format(
                global_block, global_block+cpu_block.shape[1], block_path)
            )

    def capture_kv(
        self, global_block: int, kv_cache,
    ):
        kv_record = {}
        for layer_id in self.capture_kv_layers:
            cache = kv_cache[layer_id]
            start, end = cache["write_info"]
            new_k = cache["k"][:, start: end]
            new_v = cache["v"][:, start: end]
            kv_record[layer_id] = {
                "k_hist": new_k.detach().cpu(),
                "v_hist": new_v.detach().cpu(),
                "write_info": (start, end)
            }
            print("Layer: ", layer_id, "k: ", new_k.shape, "v: ", new_v.shape)
        kv_path = os.path.join(self.output_dir, f"kv_{global_block}.pt")
        torch.save(kv_record, kv_path)
    
    def save_noise_once(self, noise):
        noise_path = os.path.join(self.output_dir, f"starting_noise.pt")
        if not os.path.exists(noise_path):
            print("[LatentCollector] Initial noise saved")
            torch.save(noise.detach().to("cpu"), noise_path)

    def finalize(self) -> str:
        if not self.enabled:
            return ""

