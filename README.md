radar_jepa/
├── main.py                          # CLI entry point (train/eval/stress/infer)
├── setup.py                         # pip-installable package
├── requirements.txt                 # Dependencies
├── configs/
│   └── default.yaml                 # All hyperparameters in one place
├── dataset/
│   ├── __init__.py
│   ├── nuscenes_dataset.py          # nuScenes front cam + radar loader
│   └── transforms.py                # Normalize, resize, stress augmentations
├── models/
│   ├── __init__.py
│   ├── backbones.py                 # ImageEncoder (ResNet18), RadarEncoder (MLP)
│   ├── heads.py                     # DetectionHead, VelocityHead, TrajectoryHead
│   └── jepa.py                      # JEPA with EMA target encoder + predictor
├── cuda/
│   ├── __init__.py                  # JIT loader + PyTorch fallbacks for all 3 kernels
│   ├── setup_cuda.py                # setuptools build script
│   └── csrc/
│       ├── bindings.cpp             # Pybind11 module exposing all 3 ops
│       ├── radar_projection.cu      # 3D→2D projection kernel
│       ├── bev_voxelize.cu          # BEV grid accumulation (atomicAdd)
│       └── radar_rasterize.cu       # 2D canvas rasterization (atomicAdd, circular splat)
├── training/
│   ├── __init__.py
│   ├── losses.py                    # JEPA latent MSE + detection + velocity + trajectory losses
│   └── trainer.py                   # AMP, grad accumulation, EMA, DataParallel, checkpoints
├── eval/
│   ├── __init__.py
│   ├── metrics.py                   # mAP, ATE, ASE, AVE, NDS, velocity error, minADE, minFDE, MissRate
│   ├── evaluator.py                 # Full benchmark pipeline (multi-GPU inference)
│   └── stress_test.py               # Robustness under low-light, fog, occlusion
└── inference/
    ├── __init__.py
    └── realtime.py                  # OpenCV loop with bbox + velocity arrow + trajectory overlay
TODO Before training:
1. ADD CNN to detection and velocity heads
2. Validate jepa code
3. Validate voxelization and rasterization and projection code on a few frames.

    EXPERIMENTS TO RUN:
    1. This from scratch jepa based training
    2. Use encoder from VJEPA and then do the following:
        Stage 1 — Load pretrained image encoder
        use weights from V-JEPA backbone
        ignore predictor head
        Stage 2 — Freeze partially

        Start with:

        freeze early layers
        train later layers + heads

        Then:

        gradually unfreeze
        Stage 3 — Add radar branch
        BEV → CNN → features
