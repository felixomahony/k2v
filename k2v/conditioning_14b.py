"""Keyframe conditioning for Wan2.2-I2V-A14B.

Wan's image-to-video model is conditioned on `y = concat(mask, vae_encode(pixels))`:
a clip of the same length as the output in which frame 0 is the input image and every
other frame is unobserved (mid-grey), plus a mask marking which frames were given. The
stock pipeline only ever marks frame 0. Nothing in the architecture requires that, so
K2V writes each keyframe into its own frame of the clip and marks that frame in the
mask. The rest of generation is Wan's own sampler, unchanged.

Frames are addressed in latents of 4 (latent 0 = frame 0, latent j = frames 4j-3..4j),
which is why keyframe times snap to every 4th frame.
"""

from __future__ import annotations

import math
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


def load(ckpt_dir: Path):
    import wan
    from wan.configs import WAN_CONFIGS
    return wan.WanI2V(config=WAN_CONFIGS["i2v-A14B"], checkpoint_dir=str(ckpt_dir),
                      device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False,
                      t5_cpu=False, convert_model_dtype=True)


def conditioning(pipe, first, keys: dict[int, torch.Tensor], F, h, w, lat_h, lat_w):
    """The (20, latent frames, lat_h, lat_w) tensor y: 4 mask channels + 16 latent."""
    msk = torch.ones(1, F, lat_h, lat_w, device=pipe.device)
    msk[:, 1:] = 0
    by_latent = {}
    for p, img in keys.items():
        by_latent[-(-p // 4)] = (p, img)          # the latent frame that holds pixel p
        msk[:, p] = 1                             # mark the keyframe's own frame only
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
                       dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]

    def resize(img):
        return torch.nn.functional.interpolate(img[None].cpu(), size=(h, w),
                                               mode="bicubic").transpose(0, 1)

    pixels = torch.concat([resize(first), torch.zeros(3, F - 1, h, w)], dim=1)
    for p, img in by_latent.values():
        pixels[:, p] = resize(img)[:, 0]          # the keyframe in its own frame only
    y = pipe.vae.encode([pixels.to(pipe.device)])[0]
    return torch.concat([msk, y])


def generate(pipe, first_frame: Path, caption: str, frames: dict[int, Path], *,
             seed: int, out_path: Path, negative_prompt: str, size: str, frame_num: int,
             fps: int, steps: int, guide_scale: float, shift: float) -> Path:
    from wan.configs import MAX_AREA_CONFIGS
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from wan.utils.utils import save_video

    def load_image(path):
        return TF.to_tensor(Image.open(path).convert("RGB")).sub_(0.5).div_(0.5).to(pipe.device)

    first = load_image(first_frame)
    keys = {p: load_image(path) for p, path in frames.items()}

    # Output size and sequence length, exactly as WanI2V.generate computes them.
    F, max_area = frame_num, MAX_AREA_CONFIGS[size]
    h, w = first.shape[1:]
    lat_h = round(np.sqrt(max_area * h / w) // pipe.vae_stride[1] //
                  pipe.patch_size[1] * pipe.patch_size[1])
    lat_w = round(np.sqrt(max_area / (h / w)) // pipe.vae_stride[2] //
                  pipe.patch_size[2] * pipe.patch_size[2])
    h, w = lat_h * pipe.vae_stride[1], lat_w * pipe.vae_stride[2]
    seq_len = ((F - 1) // pipe.vae_stride[0] + 1) * lat_h * lat_w // (
        pipe.patch_size[1] * pipe.patch_size[2])
    seq_len = int(math.ceil(seq_len / pipe.sp_size)) * pipe.sp_size

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=pipe.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(16, (F - 1) // pipe.vae_stride[0] + 1, lat_h, lat_w,
                        dtype=torch.float32, generator=seed_g, device=pipe.device)

    pipe.text_encoder.model.to(pipe.device)
    context = pipe.text_encoder([caption], pipe.device)
    context_null = pipe.text_encoder([negative_prompt or pipe.sample_neg_prompt], pipe.device)

    y = conditioning(pipe, first, keys, F, h, w, lat_h, lat_w)

    @contextmanager
    def noop():
        yield

    guide = (guide_scale, guide_scale)
    with (torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad(),
          getattr(pipe.low_noise_model, "no_sync", noop)(),
          getattr(pipe.high_noise_model, "no_sync", noop)()):
        boundary = pipe.boundary * pipe.num_train_timesteps
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=pipe.num_train_timesteps,
                                                shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(steps, device=pipe.device, shift=shift)

        latent = noise
        arg_c = {"context": [context[0]], "seq_len": seq_len, "y": [y]}
        arg_null = {"context": context_null, "seq_len": seq_len, "y": [y]}
        for t in tqdm(scheduler.timesteps):
            model = pipe._prepare_model_for_timestep(t, boundary, False)
            scale = guide[1] if t.item() >= boundary else guide[0]
            timestep = torch.stack([t]).to(pipe.device)
            cond = model([latent.to(pipe.device)], t=timestep, **arg_c)[0]
            uncond = model([latent.to(pipe.device)], t=timestep, **arg_null)[0]
            pred = uncond + scale * (cond - uncond)
            latent = scheduler.step(pred.unsqueeze(0), t, latent.unsqueeze(0),
                                    return_dict=False, generator=seed_g)[0].squeeze(0)
        video = pipe.vae.decode([latent])[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(tensor=video[None], save_file=str(out_path), fps=fps, nrow=1,
               normalize=True, value_range=(-1, 1))
    if not out_path.exists():
        raise RuntimeError(f"Wan wrote nothing to {out_path}")
    return out_path
