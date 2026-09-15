#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from geocohuman.cpif import CPIFLoss
from geocohuman.data import ImplicitReconstructionDataset
from geocohuman.geometry import safe_normalize
from geocohuman.model import GeoCoHumanImplicitModel
from human_depth_diffusion.utils import choose_device, move_batch, save_checkpoint, seed_everything


QUERY_KEYS = (
    "query_points",
    "calibration",
    "smpl_vertices",
    "smpl_faces",
    "face_neighbors",
    "face_neighbor_mask",
    "point_positions",
    "point_normals",
    "smpl_vertex_semantics",
    "point_mask",
    "nearest_face_indices",
    "point_neighbor_indices",
)


def build_dataset(config, split: str) -> ImplicitReconstructionDataset:
    data = config["data"]
    return ImplicitReconstructionDataset(
        data[f"{split}_manifest"],
        image_size=(data["image_height"], data["image_width"]),
        num_queries=data["num_queries"],
        num_prior_points=data["num_prior_points"],
        face_rings=data.get("face_rings", 2),
        max_face_neighbors=data.get("max_face_neighbors", 32),
        training=split == "train",
    )


def initialize_kaiming(module: torch.nn.Module) -> None:
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def predict(model, batch, loss_cfg, training: bool):
    query_inputs = {key: batch[key] for key in QUERY_KEYS if key in batch}
    normal_available = "normal_target" in batch and float(loss_cfg["normal_weight"]) > 0
    normal_mode = loss_cfg.get("normal_mode", "finite_difference")
    if normal_available and normal_mode == "finite_difference":
        return model.predict_with_normals(
            batch["image"],
            normal_epsilon=float(loss_cfg.get("normal_epsilon", 1e-3)),
            recompute_nearest_for_normals=bool(
                loss_cfg.get("recompute_nearest_for_normals", False)
            ),
            **query_inputs,
        )
    if normal_available and normal_mode == "autograd":
        query_points = query_inputs["query_points"].requires_grad_(True)
        query_inputs["query_points"] = query_points
        prediction = model(batch["image"], **query_inputs)
        gradient = torch.autograd.grad(
            prediction["sdf"].sum(),
            query_points,
            create_graph=training,
            retain_graph=training,
        )[0]
        prediction["normal"] = safe_normalize(gradient)
        return prediction
    return model(batch["image"], **query_inputs)


def run_epoch(model, criterion, loader, device, loss_cfg, optimizer=None, gradient_clip=1.0):
    training = optimizer is not None
    model.train(training)
    totals, count = {}, 0
    progress = tqdm(loader, desc="train" if training else "val")
    for batch in progress:
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        grad_context = torch.enable_grad() if training or loss_cfg.get("normal_mode") == "autograd" else torch.no_grad()
        with grad_context:
            prediction = predict(model, batch, loss_cfg, training)
            loss, logs = criterion(
                prediction,
                batch["occupancy"],
                batch["sdf"],
                batch.get("normal_target"),
                batch.get("normal_mask"),
            )
        if training:
            loss.backward()
            clip_grad_norm_(model.parameters(), float(gradient_clip))
            optimizer.step()
        batch_size = batch["image"].shape[0]
        count += batch_size
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        progress.set_postfix(loss=f"{logs['loss']:.4f}")
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GeoCoHuman PFA + CPIF")
    parser.add_argument("--config", default="configs/geocohuman.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(config["train"]["seed"]))
    device = choose_device(args.device)
    print(f"device={device}")

    train_data, val_data = build_dataset(config, "train"), build_dataset(config, "val")
    loader_kwargs = dict(
        batch_size=int(config["train"]["batch_size"]),
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **loader_kwargs)

    model = GeoCoHumanImplicitModel(**config["model"]).to(device)
    model.apply(initialize_kaiming)
    criterion = CPIFLoss(**{
        key: config["loss"][key]
        for key in ("occupancy_weight", "sdf_weight", "normal_weight", "positive_balance")
    })
    train_cfg = config["train"]
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(map(int, train_cfg.get("lr_milestones", (4, 8)))),
        gamma=float(train_cfg.get("lr_gamma", 0.1)),
    )
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1

    output_dir = Path(train_cfg["output_dir"])
    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        train_logs = run_epoch(
            model, criterion, train_loader, device, config["loss"], optimizer,
            train_cfg.get("gradient_clip", 1.0)
        )
        val_logs = run_epoch(model, criterion, val_loader, device, config["loss"])
        scheduler.step()
        print(f"epoch={epoch} train={train_logs} val={val_logs}")
        state = {
            "epoch": epoch,
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        save_checkpoint(state, output_dir / "last.pt")
        if (epoch + 1) % int(train_cfg.get("save_every", 1)) == 0:
            save_checkpoint(state, output_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
