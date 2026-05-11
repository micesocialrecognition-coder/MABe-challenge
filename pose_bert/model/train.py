"""PoseBERT SSL pretraining loop.

Usage:
    python -m pose_bert.model.train ./data/pose_bert_processed --num_epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn

from pose_bert.model.dataset import create_pretrain_dataloaders
from pose_bert.model.metadata import META_UNK_P_DEFAULT
from pose_bert.model.masking import (
    generate_span_mask_batch,
    generate_two_disjoint_span_masks_batch,
)


MODEL_REGISTRY = {
    "pos_raw":      ("pose_bert.model.pose_bert_pos_raw",      "single_span"),
    "pos_bins":     ("pose_bert.model.pose_bert_pos_bins",     "single_span"),
    "vel_bins":     ("pose_bert.model.pose_bert_vel_bins",     "single_span"),
    "pos_vel_bins": ("pose_bert.model.pose_bert_pos_vel_bins", "two_disjoint_spans"),
    "forecast":     ("pose_bert.model.pose_bert_forecast",     "none"),
}


def _import_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown --model {name!r}. Choices: {list(MODEL_REGISTRY)}")
    module_path, mask_mode = MODEL_REGISTRY[name]
    module = __import__(module_path, fromlist=["PoseBERT", "POSEBERT_CONFIG"])
    return module.PoseBERT, module.POSEBERT_CONFIG, mask_mode


def _call_model(model, batch, device, model_name: str):
    features = batch["features"].to(device)
    meta_ids = batch["meta_ids"].to(device)
    padding_mask = batch["padding_mask"].to(device)

    if model_name in ("pos_raw", "pos_bins", "vel_bins"):
        span_mask = batch["span_mask"].to(device)
        return model(features, meta_ids, padding_mask, span_mask)
    if model_name == "pos_vel_bins":
        span_mask_pos = batch["span_mask_pos"].to(device)
        span_mask_vel = batch["span_mask_vel"].to(device)
        return model(features, meta_ids, padding_mask, span_mask_pos, span_mask_vel)
    if model_name == "forecast":
        return model(features, meta_ids, padding_mask)
    raise ValueError(f"Unknown model_name={model_name!r}")


def _call_model_val(model, batch, device, model_name: str, config: dict, val_mask_seed: int):
    """Regenerates masks deterministically (per batch index) so val loss is comparable."""
    features = batch["features"].to(device)
    meta_ids = batch["meta_ids"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    pad_cpu = batch["padding_mask"]

    if model_name in ("pos_raw", "pos_bins", "vel_bins"):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(val_mask_seed)
        span_mask = generate_span_mask_batch(
            padding_mask=pad_cpu,
            mask_ratio=config.get("mask_ratio_single", 0.15),
            min_span=config.get("min_span", 5),
            max_span=config.get("max_span", 30),
            generator=gen,
        ).to(device)
        return model(features, meta_ids, padding_mask, span_mask)

    if model_name == "pos_vel_bins":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(val_mask_seed)
        span_mask_pos, span_mask_vel = generate_two_disjoint_span_masks_batch(
            padding_mask=pad_cpu,
            mask_ratio_pos=config.get("mask_ratio_pos", 0.15),
            mask_ratio_vel=config.get("mask_ratio_vel", 0.15),
            min_span=config.get("min_span", 5),
            max_span=config.get("max_span", 30),
            generator=gen,
        )
        return model(
            features, meta_ids, padding_mask,
            span_mask_pos.to(device), span_mask_vel.to(device),
        )

    if model_name == "forecast":
        return model(features, meta_ids, padding_mask)

    raise ValueError(f"Unknown model_name={model_name!r}")


class TeeLogger:
    """Duplicates stdout to both the terminal and a log file."""

    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


def get_device() -> torch.device:
    """Pick best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _append_metric(path: str | None, record: dict) -> None:
    """Append JSON record to metrics.jsonl; fsync so preemption can't lose it."""
    if not path:
        return
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
    scaler: torch.cuda.amp.GradScaler | None = None,
    metrics_path: str | None = None,
    epoch: int = 0,
    start_time: float = 0.0,
    model_name: str = "pos_raw",
) -> tuple[float, int]:
    """Train one epoch with fresh random masks per batch; AMP when scaler provided."""
    model.train()
    total_loss = 0.0
    total_masked = 0
    num_batches = len(dataloader)
    if num_batches == 0:
        return 0.0, 0

    use_amp = scaler is not None and device.type == "cuda"

    log_points = {int(num_batches * p) for p in (0.25, 0.5, 0.75)} - {0}

    for batch_idx, batch in enumerate(dataloader):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = _call_model(model, batch, device, model_name)
            loss = output["loss"]
        features = batch["features"]

        optimizer.zero_grad()
        if use_amp:
            # Scale loss to prevent float16 underflow, then unscale before clip.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_masked += output["num_masked"]

        if batch_idx in log_points:
            pct = int(100 * (batch_idx + 1) / num_batches)
            running_loss = total_loss / ((batch_idx + 1) * dataloader.batch_size)
            print(f"  [{pct:>3d}%] loss: {running_loss:.4f}")
            _append_metric(metrics_path, {
                "type": "step",
                "epoch": epoch + 1,
                "progress": round((batch_idx + 1) / num_batches, 3),
                "train_loss": running_loss,
                "time_sec": round(time.time() - start_time, 2),
            })

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, total_masked


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict,
    scaler: torch.cuda.amp.GradScaler | None = None,
    model_name: str = "pos_raw",
) -> tuple[float, int]:
    """Validate with a fixed mask seed so val loss is comparable across epochs."""
    model.eval()
    total_loss = 0.0
    total_masked = 0
    use_amp = scaler is not None and device.type == "cuda"

    if len(dataloader.dataset) == 0:
        return 0.0, 0

    val_seed = int(config.get("seed", 42))

    for batch_idx, batch in enumerate(dataloader):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = _call_model_val(
                model, batch, device, model_name, config,
                val_mask_seed=val_seed + batch_idx,
            )
            loss = output["loss"]

        bsz = batch["features"].size(0)
        total_loss += loss.item() * bsz
        total_masked += output.get("num_masked", 0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, total_masked


def train(
    data_dir: str,
    output_dir: str = "checkpoints/pose_bert",
    num_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_workers: int = 0,
    max_videos: int = 0,
    val_split: float = 0.1,
    seed: int = 42,
    in_memory: bool = True,
    # Model config (allow overriding defaults from POSEBERT_CONFIG)
    d_model: int = 256,
    num_layers: int = 6,
    nhead: int = 8,
    window_size: int = 128,
    stride: int = 64,
    mask_ratio_single: float = 0.15,
    min_span: int = 5,
    max_span: int = 30,
    model_name: str = "pos_raw",
    mask_ratio_pos: float = 0.15,
    mask_ratio_vel: float = 0.15,
    metadata_csv: Optional[str] = None,
    meta_unk_p: float = META_UNK_P_DEFAULT,
    augment: bool = False,
    aug_flip_p: float = 0.5,
    aug_rot_max_deg: float = 30.0,
    aug_coord_jitter_std: float = 0.3,
) -> nn.Module:
    """PoseBERT SSL pretraining loop; returns the model with best-val weights loaded."""
    start_time = time.time()
    device = get_device()
    print(f"Device: {device}")

    PoseBERT, POSEBERT_CONFIG, mask_mode = _import_model(model_name)
    print(f"Model variant: {model_name}  (mask_mode={mask_mode})")

    config = {
        **POSEBERT_CONFIG,
        "d_model": d_model,
        "num_layers": num_layers,
        "nhead": nhead,
        "window_size": window_size,
        "stride": stride,
        "mask_ratio_single": mask_ratio_single,
        "min_span": min_span,
        "max_span": max_span,
        "batch_size": batch_size,
        "lr": lr,
        "num_epochs": num_epochs,
        "seed": seed,
        "val_split": val_split,
        "model_name": model_name,
        "mask_mode": mask_mode,
        "mask_ratio_pos": mask_ratio_pos,
        "mask_ratio_vel": mask_ratio_vel,
        "metadata_csv": metadata_csv,
        "meta_unk_p": meta_unk_p,
        "augment": augment,
        "aug_flip_p": aug_flip_p,
        "aug_rot_max_deg": aug_rot_max_deg,
        "aug_coord_jitter_std": aug_coord_jitter_std,
    }

    train_loader, val_loader = create_pretrain_dataloaders(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed,
        max_videos=max_videos,
        in_memory=in_memory,
        mask_ratio_single=mask_ratio_single,
        min_span=min_span,
        max_span=max_span,
        mask_mode=mask_mode,
        mask_ratio_pos=mask_ratio_pos,
        mask_ratio_vel=mask_ratio_vel,
        metadata_csv=metadata_csv,
        meta_unk_p=meta_unk_p,
        augment=augment,
        aug_flip_p=aug_flip_p,
        aug_rot_max_deg=aug_rot_max_deg,
        aug_coord_jitter_std=aug_coord_jitter_std,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    has_val = n_val > 0
    print(f"Data: {n_train} train windows, {n_val} val windows")

    # input_dim derives from data (3 * n_parts) so V1/V2 work without a separate flag.
    config["meta_vocab_sizes"] = list(train_loader.dataset.meta_vocab_sizes)
    config["input_dim"] = int(train_loader.dataset.feature_dim)

    # Stash parts schema (if preprocessing wrote one) so checkpoints carry body-part list.
    parts_meta_path = os.path.join(data_dir, "parts.json")
    if os.path.exists(parts_meta_path):
        with open(parts_meta_path) as f:
            config["parts_schema"] = json.load(f)

    print(f"meta_vocab_sizes: {config['meta_vocab_sizes']}")
    print(f"input_dim: {config['input_dim']}  (from dataset; n_parts={config['input_dim']//3})")
    print(f"Config: {json.dumps({k: v for k, v in config.items()}, indent=2, default=str)}")

    model = PoseBERT(config).to(device)
    model = torch.compile(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {trainable_params:,} trainable params ({total_params:,} total)")

    # Floor LR at 1% of base so the last few epochs still receive meaningful gradient.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01,
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    print(f"AMP: {'enabled (float16)' if scaler is not None else 'disabled'}")

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = -1
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    config_path = os.path.join(output_dir, "config.json")
    metrics_path = os.path.join(output_dir, "metrics.jsonl")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Config saved to {config_path}")

    open(metrics_path, "w").close()
    print(f"Metrics log: {metrics_path}")

    print("=" * 60)
    for epoch in range(num_epochs):
        train_loss, train_masked = train_one_epoch(
            model, train_loader, optimizer, device, config, scaler=scaler,
            metrics_path=metrics_path, epoch=epoch, start_time=start_time,
            model_name=model_name,
        )

        val_loss, val_masked = validate(
            model, val_loader, device, config, scaler=scaler,
            model_name=model_name,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        total_masked = train_masked + val_masked

        val_str = f"Val loss: {val_loss:.4f}" if has_val else "Val: (none)"
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"{val_str} | "
            f"Masked: {_format_count(total_masked)} frames | "
            f"LR: {current_lr:.2e}"
        )

        improved = False
        if has_val:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                improved = True
        else:
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_epoch = epoch + 1
                improved = True

        if improved:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "val_loss": best_val_loss,
                },
                checkpoint_path,
            )
            metric_name = "val_loss" if has_val else "train_loss"
            print(f"  -> Saved best model ({metric_name}={best_val_loss:.4f})")

        _append_metric(metrics_path, {
            "type": "epoch",
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss if has_val else None,
            "lr": current_lr,
            "train_masked": train_masked,
            "val_masked": val_masked,
            "time_sec": round(time.time() - start_time, 2),
            "improved": improved,
        })

    elapsed = time.time() - start_time
    minutes = elapsed / 60
    print("=" * 60)
    print("Training complete!")
    print(f"  Total time:      {minutes:.1f} min")
    print(f"  Best val loss:   {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  Total params:    {total_params:,}")
    print(f"  Trainable:       {trainable_params:,}")
    print(f"  Train windows:   {n_train:,}")
    print(f"  Val windows:     {n_val:,}")
    print(f"  Checkpoint:      {checkpoint_path}")
    print(f"  Config:          {config_path}")

    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"  Loaded best weights from epoch {best_ckpt['epoch']}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoseBERT SSL Pretraining")
    parser.add_argument(
        "data_dir", nargs="?", default="./data/pose_bert_processed",
        help="Directory with index.csv and per-mouse .npy files",
    )
    parser.add_argument("--output_dir", default="checkpoints/pose_bert")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=0, help="Limit to N videos (0=all).")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in_memory", action="store_true", default=True)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument(
        "--mask_ratio_single", type=float, default=0.30,
        help="Span-mask ratio for single-span variants (pos_raw, pos_bins, vel_bins).",
    )
    parser.add_argument("--min_span", type=int, default=5, help="Min span length for span masking.")
    parser.add_argument("--max_span", type=int, default=30, help="Max span length for span masking.")
    parser.add_argument(
        "--model", default="pos_raw",
        choices=list(MODEL_REGISTRY.keys()),
        help="Pretraining variant: pos_raw, pos_bins, vel_bins, pos_vel_bins, forecast.",
    )
    parser.add_argument("--mask_ratio_pos", type=float, default=0.15,
        help="Position mask ratio for pos_vel_bins.")
    parser.add_argument("--mask_ratio_vel", type=float, default=0.15,
        help="Velocity mask ratio for pos_vel_bins.")
    parser.add_argument("--run_name", default=None,
        help="Run folder name under {output_dir}/runs/. Default: timestamp.")
    parser.add_argument("--metadata_csv", default=None,
        help="Path to data/raw/train.csv for metadata conditioning.")
    parser.add_argument("--meta_unk_p", type=float, default=META_UNK_P_DEFAULT,
        help=f"Per-field random UNK injection rate (default {META_UNK_P_DEFAULT}). Val uses 0.")
    parser.add_argument("--augment", action="store_true",
        help="Enable pose augmentation on training set (requires {data_dir}/parts.json).")
    parser.add_argument("--aug_flip_p", type=float, default=0.5,
        help="Prob of horizontal flip + L/R part swap per sample.")
    parser.add_argument("--aug_rot_max_deg", type=float, default=30.0,
        help="Max rotation magnitude in degrees, around the observed centroid.")
    parser.add_argument("--aug_coord_jitter_std", type=float, default=0.3,
        help="Per-coord Gaussian jitter std in cm.")
    args = parser.parse_args()

    # Per-run layout: {output_dir}/runs/{run_name}/ with a "latest" symlink.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or timestamp
    run_dir = os.path.join(args.output_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    latest_link = os.path.join(args.output_dir, "latest")
    if os.path.islink(latest_link) or os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(os.path.relpath(run_dir, args.output_dir), latest_link)

    log_path = os.path.join(run_dir, "train.log")
    tee = TeeLogger(log_path)
    sys.stdout = tee
    print(f"Run folder: {run_dir}")
    print(f"Logging to {log_path}")
    print(f"Command: {' '.join(sys.argv)}")
    print(f"Started: {datetime.now().isoformat()}")
    print("-" * 60)

    try:
        train(
            data_dir=args.data_dir,
            output_dir=run_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            max_videos=args.max_videos,
            val_split=args.val_split,
            seed=args.seed,
            in_memory=args.in_memory,
            d_model=args.d_model,
            num_layers=args.num_layers,
            nhead=args.nhead,
            window_size=args.window_size,
            stride=args.stride,
            mask_ratio_single=args.mask_ratio_single,
            min_span=args.min_span,
            max_span=args.max_span,
            model_name=args.model,
            mask_ratio_pos=args.mask_ratio_pos,
            mask_ratio_vel=args.mask_ratio_vel,
            metadata_csv=args.metadata_csv,
            meta_unk_p=args.meta_unk_p,
            augment=args.augment,
            aug_flip_p=args.aug_flip_p,
            aug_rot_max_deg=args.aug_rot_max_deg,
            aug_coord_jitter_std=args.aug_coord_jitter_std,
        )
    finally:
        print("-" * 60)
        print(f"Finished: {datetime.now().isoformat()}")
        tee.close()
