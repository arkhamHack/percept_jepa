"""Entry point for multimodal perception pipeline.

Usage:
    python main.py --mode train   --config configs/default.yaml
    python main.py --mode eval    --config configs/default.yaml
    python main.py --mode stress  --config configs/default.yaml
    python main.py --mode infer   --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
import torch
from torch.utils.data import DataLoader

from dataset import NuScenesRadarCameraDataset, collate_fn
from models import MultimodalPerceptionModel, JEPAModel
from training import Trainer


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    mc = cfg["model"]
    model_type = mc.get("type", "multimodal_vjepa")

    if model_type == "multimodal_vjepa":
        return MultimodalPerceptionModel(
            vjepa_cfg=mc.get("vjepa", {}),
            radar_cfg=mc.get("radar", {}),
            fusion_cfg=mc.get("fusion", {}),
            detection_cfg=mc.get("detection", {}),
            velocity_cfg=mc.get("velocity", {}),
            tracking_cfg=mc.get("tracking", {}),
            bev_cfg=cfg.get("bev", {}),
        )
    else:
        # Legacy JEPA model
        return JEPAModel(
            pretrained=mc.get("pretrained", True),
            light=mc.get("light_backbone", False),
            image_feat_dim=mc.get("image_feat_dim", 512),
            radar_dim=mc.get("radar_dim", 128),
            latent_dim=mc.get("latent_dim", 256),
            max_objects=mc.get("max_objects", 50),
            num_modes=mc.get("num_modes", 5),
            pred_steps=mc.get("pred_steps", 12),
            agent_state_dim=mc.get("agent_state_dim", 3),
            decoder_heads=mc.get("decoder_heads", 8),
            decoder_layers=mc.get("decoder_layers", 2),
        )


def build_dataloaders(
    cfg: dict[str, Any],
    splits: list[str] = ("train", "val"),
) -> dict[str, DataLoader]:
    dc = cfg["dataset"]
    tc = cfg["training"]
    pc = cfg.get("prediction", {})
    loaders: dict[str, DataLoader] = {}

    for split in splits:
        ds = NuScenesRadarCameraDataset(
            nuscenes_root=dc["nuscenes_root"],
            version=dc.get("version", "v1.0-mini"),
            split=split,
            image_size=tuple(dc.get("image_size", [448, 800])),
            max_radar_points=dc.get("max_radar_points", 256),
            use_future_frame=True,
            prediction_seconds=pc.get("seconds", 6.0),
            prediction_hz=pc.get("hz", 2),
        )
        bs = tc["batch_size"] if split == "train" else cfg["eval"].get("batch_size", tc["batch_size"])
        loaders[split] = DataLoader(
            ds,
            batch_size=bs,
            shuffle=(split == "train"),
            num_workers=tc.get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    return loaders


def build_optimizer_scheduler(
    model: torch.nn.Module,
    cfg: dict[str, Any],
) -> tuple[torch.optim.Optimizer, Any]:
    tc = cfg["training"]
    sc = cfg.get("scheduler", {})

    # Staged training: separate param groups for backbone vs rest
    if hasattr(model, "get_backbone_params"):
        backbone_lr = tc.get("lr", 3e-4) * tc.get("backbone_lr_scale", 0.1)
        param_groups = [
            {"params": model.get_non_backbone_params(), "lr": tc.get("lr", 3e-4)},
            {"params": [p for p in model.get_backbone_params() if p.requires_grad],
             "lr": backbone_lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": tc.get("lr", 3e-4)}]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=tc.get("weight_decay", 1e-4),
    )

    sched_type = sc.get("type", "cosine")
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sc.get("T_max", tc.get("epochs", 50)),
        )
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sc.get("step_size", 15),
            gamma=sc.get("gamma", 0.1),
        )
    else:
        scheduler = None

    return optimizer, scheduler


# ──────────────────────────────────────────────────────────────
#  Mode handlers
# ──────────────────────────────────────────────────────────────

def run_train(cfg: dict[str, Any]) -> None:
    model = build_model(cfg)

    # Set training stage for multimodal model
    tc = cfg["training"]
    stage = tc.get("stage", 1)
    if hasattr(model, "set_training_stage"):
        unfreeze_n = cfg["model"].get("vjepa", {}).get("unfreeze_last_n", 4)
        model.set_training_stage(stage, unfreeze_n)
        print(f"[main] Training stage {stage}" +
              (f" (unfreezing last {unfreeze_n} V-JEPA blocks)" if stage >= 2 else " (V-JEPA frozen)"))

    loaders = build_dataloaders(cfg, splits=["train", "val"])
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)

    tc = cfg["training"]
    trainer_cfg = {
        "epochs": tc.get("epochs", 50),
        "grad_accum_steps": tc.get("grad_accum_steps", 1),
        "amp_enabled": tc.get("amp_enabled", True),
        "ema_momentum": tc.get("ema_momentum", 0.996),
        "checkpoint_dir": tc.get("checkpoint_dir", "checkpoints"),
        "log_interval": tc.get("log_interval", 50),
        "loss_weights": tc.get("loss_weights"),
    }

    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        config=trainer_cfg,
    )

    resume = tc.get("resume_checkpoint")
    if resume and Path(resume).exists():
        trainer.load_checkpoint(resume)

    trainer.train()


def run_eval(cfg: dict[str, Any]) -> None:
    from eval import Evaluator

    model = build_model(cfg)
    ckpt_path = cfg["inference"].get("checkpoint", "checkpoints/best.pt")
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[main] Loaded checkpoint from {ckpt_path}")
    else:
        print(f"[main] WARNING: checkpoint not found at {ckpt_path}, using random weights")

    loaders = build_dataloaders(cfg, splits=["val"])
    ec = cfg.get("eval", {})

    evaluator = Evaluator(
        model=model,
        dataloader=loaders["val"],
        iou_thresholds=ec.get("iou_thresholds", [0.3, 0.5, 0.7]),
        multi_gpu=ec.get("multi_gpu", True),
        amp_enabled=cfg["training"].get("amp_enabled", True),
    )
    evaluator.evaluate()


def run_stress(cfg: dict[str, Any]) -> None:
    from eval import StressTestRunner

    model = build_model(cfg)
    ckpt_path = cfg["inference"].get("checkpoint", "checkpoints/best.pt")
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    loaders = build_dataloaders(cfg, splits=["val"])
    sc = cfg.get("stress_test", {})

    runner = StressTestRunner(
        model=model,
        dataloader=loaders["val"],
        config=sc,
        iou_thresholds=cfg.get("eval", {}).get("iou_thresholds", [0.3, 0.5, 0.7]),
    )
    runner.run()


def run_infer(cfg: dict[str, Any]) -> None:
    from inference import RealtimeInference

    model = build_model(cfg)
    ic = cfg.get("inference", {})
    ckpt_path = ic.get("checkpoint", "checkpoints/best.pt")
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    runner = RealtimeInference(
        model=model,
        image_size=tuple(ic.get("image_size", [448, 800])),
        confidence_threshold=ic.get("confidence_threshold", 0.3),
        show_velocity=ic.get("show_velocity", True),
    )
    runner.run_video(
        source=ic.get("video_source", 0),
        save_path=ic.get("save_output"),
    )


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Radar-Camera Fusion JEPA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["train", "eval", "stress", "infer"],
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    mode_map = {
        "train": run_train,
        "eval": run_eval,
        "stress": run_stress,
        "infer": run_infer,
    }
    mode_map[args.mode](cfg)


if __name__ == "__main__":
    main()
