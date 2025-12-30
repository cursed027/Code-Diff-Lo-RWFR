import os
from argparse import ArgumentParser
import copy

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from diffbir.model import ControlLDM, SwinIR, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler

def get_lora_state_dict(model):
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state[name] = param.detach().cpu()
    return lora_state


def main(args) -> None:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # ----------------------------
    # Experiment directory
    # ----------------------------
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True) 
        print(f"Experiment directory created at {exp_dir}")

    # ----------------------------
    # Model
    # ----------------------------
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)

    if accelerator.is_main_process:
        print(f"Loaded SD weights | unused: {len(unused)} | missing: {len(missing)}")

    if cfg.train.resume:
        # 🔁 Resume LoRA
        cldm.load_lora(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_main_process:
            print(f"Resumed LoRA from {cfg.train.resume}")
    else:
        cldm.load_controlnet_from_unet()
        cldm.inject_lora(
            rank_unet=cfg.train.lora_rank_unet,
            rank_controlnet=cfg.train.lora_rank_controlnet,
        )

    # Freeze everything except LoRA
    for n, p in cldm.named_parameters():
        p.requires_grad = "lora_" in n

    # ----------------------------
    # SwinIR (frozen)
    # ----------------------------
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = torch.load(cfg.train.swinir_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k.replace("module.", "")): v for k, v in sd.items()}
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # ----------------------------
    # Optimizer (LoRA only)
    # ----------------------------
    unet_lora_params = []
    controlnet_lora_params = []

    for name, param in cldm.named_parameters():
        if not param.requires_grad:
            continue
        if "unet" in name:
            unet_lora_params.append(param)
        elif "controlnet" in name:
            controlnet_lora_params.append(param)

    opt = torch.optim.AdamW(
        [
            {"params": unet_lora_params, "lr": cfg.train.lr_unet},
            {"params": controlnet_lora_params, "lr": cfg.train.lr_controlnet},
        ],
        weight_decay=1e-2,
    )


    # ----------------------------
    # Dataset
    # ----------------------------
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    cldm.train().to(device)
    swinir.eval().to(device)
    diffusion.to(device)

    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm = accelerator.unwrap_model(cldm)

    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )

    writer = SummaryWriter(cfg.train.exp_dir) if accelerator.is_main_process else None

    global_step = 0
    max_steps = cfg.train.train_steps
    noise_aug_timestep = cfg.train.noise_aug_timestep
    step_loss, epoch_loss = [], []

    # ----------------------------
    # Training loop
    # ----------------------------
    while global_step < max_steps:
        pbar = tqdm(
            loader,
            disable=not accelerator.is_main_process,
            desc=f"Step {global_step}",
        )

        for batch in pbar:
            to(batch, device)
            gt, lq, cf, prompt = batch

            gt = rearrange(gt, "b h w c -> b c h w").float()
            lq = rearrange(lq, "b h w c -> b c h w").float()
            cf = rearrange(cf, "b h w c -> b c h w").float()

            with torch.no_grad():
                z0 = pure_cldm.vae_encode(gt)
                clean = swinir(lq)
                cond = pure_cldm.prepare_condition(
                    clean_img=clean, cf_img=cf, txt=prompt
                )
                cond_aug = copy.deepcopy(cond)
                if noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        cond_aug["c_img"],
                        torch.randint(
                            0, noise_aug_timestep, (z0.size(0),), device=device
                        ),
                        torch.randn_like(cond_aug["c_img"]),
                    )

            t = torch.randint(0, diffusion.num_timesteps, (z0.size(0),), device=device)
            loss = diffusion.p_losses(cldm, z0, t, cond_aug)

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())

            pbar.set_description(f"step={global_step} loss={loss.item():.4f}")

            if writer and global_step % cfg.train.log_every == 0:
                writer.add_scalar("loss/step", sum(step_loss) / len(step_loss), global_step)
                step_loss.clear()

            if writer and global_step % cfg.train.ckpt_every == 0:
                torch.save(
                    get_lora_state_dict(pure_cldm),
                    os.path.join(ckpt_dir, f"lora_{global_step:07d}.pt")
                )

            if global_step >= max_steps:
                break

        if writer:
            writer.add_scalar(
                "loss/epoch", sum(epoch_loss) / len(epoch_loss), global_step
            )
        epoch_loss.clear()

    if writer:
        writer.close()
    if accelerator.is_main_process:
        print("Training finished.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
