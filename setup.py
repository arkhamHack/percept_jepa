"""Package setup for radar_jepa."""

from setuptools import setup, find_packages

setup(
    name="radar_jepa",
    version="0.1.0",
    description="Radar-Camera Fusion with JEPA for nuScenes",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "nuscenes-devkit>=1.1.11",
        "pyquaternion>=0.9.9",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ],
)
