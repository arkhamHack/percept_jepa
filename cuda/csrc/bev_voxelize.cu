#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace radar_jepa {

__global__ void bev_voxelize_kernel(
    const float* __restrict__ points,   // (N, C)
    float* __restrict__ bev,            // (H, W, C)
    float x_min, float x_max,
    float y_min, float y_max,
    int H, int W, int C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = points[idx * C + 0];
    float py = points[idx * C + 1];

    if (px < x_min || px >= x_max || py < y_min || py >= y_max)
        return;

    float cell_w = (x_max - x_min) / static_cast<float>(W);
    float cell_h = (y_max - y_min) / static_cast<float>(H);

    int col = static_cast<int>((px - x_min) / cell_w);
    int row = static_cast<int>((py - y_min) / cell_h);

    col = min(col, W - 1);
    row = min(row, H - 1);

    for (int c = 0; c < C; ++c) {
        atomicAdd(&bev[(row * W + col) * C + c], points[idx * C + c]);
    }
}

torch::Tensor bev_voxelize_cuda(
    torch::Tensor points,
    float x_min, float x_max,
    float y_min, float y_max,
    int grid_h, int grid_w)
{
    TORCH_CHECK(points.is_cuda(),       "points must be a CUDA tensor");
    TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
    TORCH_CHECK(points.dim() == 2,      "points must be (N, C)");
    TORCH_CHECK(x_max > x_min,          "x_max must be > x_min");
    TORCH_CHECK(y_max > y_min,          "y_max must be > y_   min");

    int N = points.size(0);
    int C = points.size(1);
                                                                                                                                                    
    auto bev = torch::zeros({grid_h, grid_w, C},
                            torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(points.device()));

    if (N == 0) return bev;

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    bev_voxelize_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        bev.data_ptr<float>(),
        x_min, x_max, y_min, y_max,
        grid_h, grid_w, C, N);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return bev;
}

} // namespace radar_jepa
