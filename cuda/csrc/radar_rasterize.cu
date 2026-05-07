#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace radar_jepa {

__global__ void radar_rasterize_kernel(
    const float* __restrict__ coords_2d,  // (N, 2) projected (u, v)
    const float* __restrict__ features,   // (N, C) per-point features
    const bool*  __restrict__ valid,       // (N,) validity mask
    float* __restrict__ canvas,            // (H, W, C) output
    int N, int H, int W, int C,
    int radius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (!valid[idx]) return;

    float u = coords_2d[idx * 2 + 0];
    float v = coords_2d[idx * 2 + 1];

    int cu = __float2int_rn(u);
    int cv = __float2int_rn(v);

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int px = cu + dx;
            int py = cv + dy;

            if (px < 0 || px >= W || py < 0 || py >= H) continue;

            if (dx * dx + dy * dy > radius * radius) continue;

            for (int c = 0; c < C; ++c) {
                atomicAdd(&canvas[(py * W + px) * C + c], features[idx * C + c]);
            }
        }
    }
}

torch::Tensor radar_rasterize_cuda(
    torch::Tensor coords_2d,
    torch::Tensor features,
    torch::Tensor valid,
    int height,
    int width,
    int radius)
{
    TORCH_CHECK(coords_2d.is_cuda(),  "coords_2d must be a CUDA tensor");
    TORCH_CHECK(features.is_cuda(),   "features must be a CUDA tensor");
    TORCH_CHECK(valid.is_cuda(),      "valid must be a CUDA tensor");
    TORCH_CHECK(coords_2d.is_contiguous(), "coords_2d must be contiguous");
    TORCH_CHECK(features.is_contiguous(),  "features must be contiguous");
    TORCH_CHECK(valid.is_contiguous(),     "valid must be contiguous");

    int N = features.size(0);
    int C = features.size(1);

    auto canvas = torch::zeros({height, width, C},
                               torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(features.device()));

    if (N == 0) return canvas;

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    radar_rasterize_kernel<<<blocks, threads>>>(
        coords_2d.data_ptr<float>(),
        features.data_ptr<float>(),
        valid.data_ptr<bool>(),
        canvas.data_ptr<float>(),
        N, height, width, C, radius);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return canvas;
}

} // namespace radar_jepa
