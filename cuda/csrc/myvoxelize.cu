 #include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


torch::Tensor myvoxelize_cuda(
    torch::Tensor points,
    float x_min, float x_max,
    float y_min, float y_max,
    int grid_h, int grid_w
)
{
    TORCH_CHECK(points.is_cuda(),       "points must be a CUDA tensor");
    TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
    TORCH_CHECK(points.dim() == 2,      "points must be (N, C)");
    TORCH_CHECK(x_max > x_min,          "x_max must be > x_min");
    TORCH_CHECK(y_max > y_min,          "y_max must be > y_   min");

    
}