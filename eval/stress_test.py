"""Stress-test evaluation under adverse augmentations."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dataset.transforms import denormalize, normalize, low_light, fog, occlusion
from .evaluator import Evaluator


class StressTestRunner:
    """Evaluate JEPA model robustness under synthetic corruptions.

    Wraps :class:`Evaluator` and re-runs evaluation after applying each
    augmentation at multiple severity levels.  Reports the absolute metric
    values **and** the relative drop compared to the clean baseline.

    Args:
        model: Trained JEPAModel.
        dataloader: Validation ``DataLoader``.
        config: Stress-test configuration dict (from YAML).
        iou_thresholds: IoU thresholds forwarded to the evaluator.
    """

    _AUGMENTATIONS = {
        "low_light": {
            "fn": low_light,
            "param_key": "factors",
            "kwarg_name": "factor",
        },
        "fog": {
            "fn": fog,
            "param_key": "severities",
            "kwarg_name": "severity",
        },
        "occlusion": {
            "fn": occlusion,
            "param_key": "num_patches",
            "kwarg_name": "num_patches",
        },
    }

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        config: dict[str, Any] | None = None,
        iou_thresholds: list[float] | None = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.config = config or {}
        self.iou_thresholds = iou_thresholds or [0.3, 0.5, 0.7]

    def run(self) -> dict[str, Any]:
        """Run clean baseline + all configured augmentations.

        Returns:
            Nested dict: ``{aug_name: {severity: {metric: value, ...}, ...}}``,
            plus ``"clean"`` baseline and ``"summary"`` with relative drops.
        """
        evaluator = Evaluator(
            self.model, self.dataloader,
            iou_thresholds=self.iou_thresholds,
            multi_gpu=False,
        )

        print("\n[StressTest] === Clean baseline ===")
        clean = evaluator.evaluate()
        results: dict[str, Any] = {"clean": clean}

        aug_configs = self.config.get("augmentations", {})

        for aug_name, meta in self._AUGMENTATIONS.items():
            aug_cfg = aug_configs.get(aug_name, {})
            severities = aug_cfg.get(meta["param_key"], [])
            if not severities:
                continue

            results[aug_name] = {}
            for sev in severities:
                print(f"\n[StressTest] === {aug_name} ({meta['kwarg_name']}={sev}) ===")

                aug_loader = _AugmentedDataLoader(
                    self.dataloader,
                    aug_fn=meta["fn"],
                    aug_kwargs={meta["kwarg_name"]: sev},
                )
                aug_evaluator = Evaluator(
                    self.model, aug_loader,
                    iou_thresholds=self.iou_thresholds,
                    multi_gpu=False,
                )
                aug_results = aug_evaluator.evaluate()
                results[aug_name][sev] = aug_results

        summary = self._compute_drops(clean, results)
        results["summary"] = summary
        self._print_summary(summary)
        return results

    @staticmethod
    def _compute_drops(
        clean: dict[str, Any],
        results: dict[str, Any],
    ) -> dict[str, Any]:
        target_metrics = ["mAP", "NDS", "mAVE"]
        summary: dict[str, Any] = {}

        for aug_name, severities in results.items():
            if aug_name in ("clean", "summary") or not isinstance(severities, dict):
                continue
            summary[aug_name] = {}
            for sev, metrics in severities.items():
                if not isinstance(metrics, dict):
                    continue
                drops = {}
                for m in target_metrics:
                    if m in clean and m in metrics and clean[m] != 0:
                        drops[f"{m}_drop%"] = (1.0 - metrics[m] / clean[m]) * 100
                summary[aug_name][sev] = drops

        return summary

    @staticmethod
    def _print_summary(summary: dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("  Stress-Test Robustness Summary (% drop from clean)")
        print("=" * 60)
        for aug_name, severities in summary.items():
            for sev, drops in severities.items():
                line = f"  {aug_name} ({sev}): "
                line += "  ".join(f"{k}={v:+.1f}%" for k, v in drops.items())
                print(line)
        print("=" * 60 + "\n")


class _AugmentedDataLoader:
    """Wraps a DataLoader and applies an augmentation to the image batch.

    Operates on ImageNet-normalised tensors: denormalise → augment →
    re-normalise, so augmentations see pixel values in ``[0, 1]``.
    """

    def __init__(self, loader, aug_fn, aug_kwargs: dict | None = None):
        self.loader = loader
        self.aug_fn = aug_fn
        self.aug_kwargs = aug_kwargs or {}

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            batch = dict(batch)
            images = batch["image"]  # (B, 3, H, W) normalised

            augmented = []
            for img in images:
                raw = denormalize(img)
                aug = self.aug_fn(raw, **self.aug_kwargs)
                augmented.append(normalize(aug))
            batch["image"] = torch.stack(augmented)

            if "future_image" in batch:
                aug_future = []
                for img in batch["future_image"]:
                    raw = denormalize(img)
                    aug = self.aug_fn(raw, **self.aug_kwargs)
                    aug_future.append(normalize(aug))
                batch["future_image"] = torch.stack(aug_future)

            yield batch
