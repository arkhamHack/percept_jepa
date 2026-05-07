#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace radar_jepa {

__global__ void radar_project_kernel(
    const float* __restrict__ points,   // (N, C) with C >= 3
    const float* __restrict__ K,        // (3, 3) intrinsics
    const float* __restrict__ T,        // (4, 4) extrinsics [R|t]
    float* __restrict__ coords_2d,      // (N, 2) output
    bool* __restrict__ valid,           // (N,)   output
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float px = points[idx * 3 + 0];
    float py = points[idx * 3 + 1];
    float pz = points[idx * 3 + 2];

    // cam = R @ p + t  (extrinsics row-major 4x4)
    float cx = T[0] * px + T[1] * py + T[2]  * pz + T[3];
    float cy = T[4] * px + T[5] * py + T[6]  * pz + T[7];
    float cz = T[8] * px + T[9] * py + T[10] * pz + T[11];

    if (cz <= 0.0f) {
        coords_2d[idx * 2 + 0] = 0.0f;
        coords_2d[idx * 2 + 1] = 0.0f;
        valid[idx] = false;
        return;
    }

    // p_img = K @ cam,  then normalise by z
    float u = K[0] * cx + K[1] * cy + K[2] * cz;
    float v = K[3] * cx + K[4] * cy + K[5] * cz;
    float w = K[6] * cx + K[7] * cy + K[8] * cz;

    coords_2d[idx * 2 + 0] = u / w;
    coords_2d[idx * 2 + 1] = v / w;
    valid[idx] = true;
}

std::vector<torch::Tensor> radar_project_cuda(
    torch::Tensor points,
    torch::Tensor intrinsics,
    torch::Tensor extrinsics)
{
    TORCH_CHECK(points.is_cuda(),      "points must be a CUDA tensor");
    TORCH_CHECK(intrinsics.is_cuda(),  "intrinsics must be a CUDA tensor");
    TORCH_CHECK(extrinsics.is_cuda(),  "extrinsics must be a CUDA tensor");
    TORCH_CHECK(points.is_contiguous(),     "points must be contiguous");
    TORCH_CHECK(intrinsics.is_contiguous(), "intrinsics must be contiguous");
    TORCH_CHECK(extrinsics.is_contiguous(), "extrinsics must be contiguous");

    int N = points.size(0);

    auto opts = torch::TensorOptions().device(points.device());
    auto coords_2d = torch::zeros({N, 2}, opts.dtype(torch::kFloat32));
    auto valid     = torch::zeros({N},    opts.dtype(torch::kBool));

    if (N == 0) return {coords_2d, valid};

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    radar_project_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        intrinsics.data_ptr<float>(),
        extrinsics.data_ptr<float>(),
        coords_2d.data_ptr<float>(),
        valid.data_ptr<bool>(),
        N);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {coords_2d, valid};
}

} // namespace radar_jepa
