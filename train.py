#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from human_depth_diffusion.config import load_config
from human_depth_diffusion.data import HumanDepthDataset
from human_depth_diffusion.diffusion import GaussianDiffusion
from human_depth_diffusion.ema import EMA
from human_depth_diffusion.model import ConditionalUNet
from human_depth_diffusion.utils import choose_device, move_batch, save_checkpoint, seed_everything


def build_dataset(config, split: str) -> HumanDepthDataset:
    data = config["data"]
    return HumanDepthDataset(
        manifest=data[f"{split}_manifest"],
        image_size=(data["image_height"], data["image_width"]),
        max_depth_m=data["max_depth_m"],
        png_depth_scale=data.get("png_depth_scale", 1000.0),
        horizontal_flip=data.get("horizontal_flip", 0.0) if split == "train" else 0.0,
    )


@torch.no_grad()
def validate(model, diffusion, loader, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        loss, _ = diffusion.training_loss(
            model, batch["target"], batch["condition"], batch["target_mask"]
        )
        total += float(loss) * batch["target"].shape[0]
        count += batch["target"].shape[0]
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the conditional human-depth DDPM")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg, model_cfg, diffusion_cfg = (
        config["train"], config["model"], config["diffusion"]
    )
    seed_everything(int(train_cfg["seed"]))
    device = choose_device(args.device)
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    print(f"device={device}, amp={amp_enabled}")

    train_data = build_dataset(config, "train")
    val_data = build_dataset(config, "val")
    common_loader = dict(
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **common_loader)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **common_loader)

    model = ConditionalUNet(**model_cfg).to(device)
    diffusion = GaussianDiffusion(
        timesteps=diffusion_cfg["timesteps"],
        beta_schedule=diffusion_cfg["beta_schedule"],
        prediction_type=diffusion_cfg["prediction_type"],
        depth_l1_weight=diffusion_cfg.get("depth_l1_weight", 1.0),
        gradient_weight=diffusion_cfg.get("gradient_weight", 0.5),
    ).to(device)
    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    optimizer_class = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = optimizer_class(
        model.parameters(), lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]), betas=(0.9, 0.999)
    )
    ema = EMA(model, float(train_cfg["ema_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch, global_step = 0, 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        print(f"resumed {args.resume} at epoch {start_epoch}")

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulation = int(train_cfg.get("gradient_accumulation", 1))
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch:03d}")
        for batch_index, batch in enumerate(progress):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss, logs = diffusion.training_loss(
                    model, batch["target"], batch["condition"], batch["target_mask"]
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()

            should_step = (batch_index + 1) % accumulation == 0 or (
                batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                global_step += 1
            if batch_index % int(train_cfg["log_every"]) == 0:
                progress.set_postfix(
                    loss=f"{logs['loss']:.4f}", mse=f"{logs['diffusion_mse']:.4f}"
                )

        val_loss = None
        if (epoch + 1) % int(train_cfg["val_every"]) == 0:
            val_loss = validate(ema.model, diffusion, val_loader, device)
            print(f"epoch={epoch} val_diffusion_loss={val_loss:.6f}")

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "val_loss": val_loss,
        }
        save_checkpoint(state, output_dir / "last.pt")
        if (epoch + 1) % int(train_cfg["save_every"]) == 0:
            save_checkpoint(state, output_dir / f"epoch_{epoch:04d}.pt")


if __name__ == "__main__":
    main()
