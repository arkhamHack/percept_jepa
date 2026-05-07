"""Build script for radar_jepa CUDA extensions."""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_csrc = os.path.join(os.path.dirname(__file__), "csrc")

setup(
    name="radar_jepa_cuda",
    ext_modules=[
        CUDAExtension(
            name="radar_jepa_cuda",
            sources=[
                os.path.join(_csrc, "bindings.cpp"),
                os.path.join(_csrc, "radar_projection.cu"),
                os.path.join(_csrc, "bev_voxelize.cu"),
                os.path.join(_csrc, "radar_rasterize.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
