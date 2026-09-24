"""Keyframe conditioning for Wan2.2-TI2V-5B.

The 5B conditions by replacement rather than by a mask channel. At every denoising
step, the latent frames that are given are overwritten with the encoded image, and
those latents are handed timestep 0 while the rest get t. Stock TI2V does this for
latent 0 only; K2V does it for every keyframe, each written into the latent that holds
its frame (latent j = frames 4j-3..4j, so keyframe frames are multiples of 4).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


def load(ckpt_dir: Path):
    import wan
    from wan.configs import WAN_CONFIGS
    return wan.WanTI2V(config=WAN_CONFIGS["ti2v-5B"], checkpoint_dir=str(ckpt_dir),
                       device_id=0, rank=0, t5_cpu=False, init_on_cpu=False,
                       convert_model_dtype=True)


def _prepare(img: Image.Image, ow: int, oh: int, device) -> torch.Tensor:
    """Resize and centre-crop to the working size, as the stock pipeline does."""
    scale = max(ow / img.width, oh / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    x1, y1 = (img.width - ow) // 2, (img.height - oh) // 2
    img = img.crop((x1, y1, x1 + ow, y1 + oh))
    return TF.to_tensor(img).sub_(0.5).div_(0.5).to(device).unsqueeze(1)


def generate(pipe, first_frame: Path, caption: str, frames: dict[int, Path], *,
             seed: int, out_path: Path, negative_prompt: str, size: str, frame_num: int,
             fps: int, steps: int, guide_scale: float, shift: float,
             offload: bool = True) -> Path:
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from wan.utils.utils import best_output_size, masks_like, save_video

    cfg, device = WAN_CONFIGS["ti2v-5B"], torch.device("cuda")
    first = Image.open(first_frame).convert("RGB")
    ow, oh = best_output_size(first.width, first.height,
                              cfg.patch_size[2] * cfg.vae_stride[2],
                              cfg.patch_size[1] * cfg.vae_stride[1], MAX_AREA_CONFIGS[size])
    F = frame_num
    latent_frames = (F - 1) // cfg.vae_stride[0] + 1
    seq_len = (latent_frames * (oh // cfg.vae_stride[1]) * (ow // cfg.vae_stride[2])
               // (cfg.patch_size[1] * cfg.patch_size[2]))

    given = {0: first}                                     # latent index -> image
    for frame, path in frames.items():
        if frame % 4 or not 0 < frame // 4 < latent_frames:
            raise ValueError(f"keyframe frame {frame} is not a multiple of 4 inside the clip")
        given[frame // 4] = Image.open(path).convert("RGB")

    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)
    noise = torch.randn(pipe.vae.model.z_dim, latent_frames, oh // cfg.vae_stride[1],
                        ow // cfg.vae_stride[2], dtype=torch.float32, generator=seed_g,
                        device=device)

    pipe.text_encoder.model.to(device)
    context = pipe.text_encoder([caption], device)
    context_null = pipe.text_encoder([negative_prompt or cfg.sample_neg_prompt], device)
    if offload:
        pipe.text_encoder.model.cpu()

    # Latent 0 holds frame 0 alone, so it is a one-frame encode. Every other latent
    # holds four frames, so the keyframe is encoded as five copies and the second latent
    # taken: the one whose four frames are all the keyframe.
    z = torch.zeros_like(noise)
    for j, image in sorted(given.items()):
        pixels = _prepare(image, ow, oh, device)
        z[:, j] = (pipe.vae.encode([pixels])[0][:, 0] if j == 0 else
                   pipe.vae.encode([pixels.repeat(1, 5, 1, 1)])[0][:, 1])

    with torch.amp.autocast("cuda", dtype=cfg.param_dtype), torch.no_grad():
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps,
                                                shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(steps, device=device, shift=shift)

        _, mask2 = masks_like([noise], zero=True)
        mask = mask2[0]
        for j in given:
            mask[:, j] = 0                                  # 0 = "this latent is given"
        latent = (1. - mask) * z + mask * noise

        arg_c = {"context": [context[0]], "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}
        if offload:
            pipe.model.to(device)
            torch.cuda.empty_cache()
        for t in tqdm(scheduler.timesteps):
            timestep = torch.stack([t]).to(device)
            ts = (mask[0][:, ::2, ::2] * timestep).flatten()   # 0 on given latents
            ts = torch.cat([ts, ts.new_ones(seq_len - ts.size(0)) * timestep]).unsqueeze(0)
            cond = pipe.model([latent.to(device)], t=ts, **arg_c)[0]
            if offload:
                torch.cuda.empty_cache()
            uncond = pipe.model([latent.to(device)], t=ts, **arg_null)[0]
            if offload:
                torch.cuda.empty_cache()
            pred = uncond + guide_scale * (cond - uncond)
            latent = scheduler.step(pred.unsqueeze(0), t, latent.unsqueeze(0),
                                    return_dict=False, generator=seed_g)[0].squeeze(0)
            latent = (1. - mask) * z + mask * latent            # re-applied every step
        if offload:
            pipe.model.cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        video = pipe.vae.decode([latent])[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(tensor=video[None], save_file=str(out_path), fps=fps, nrow=1,
               normalize=True, value_range=(-1, 1))
    if not out_path.exists():
        raise RuntimeError(f"Wan wrote nothing to {out_path}")
    return out_path
