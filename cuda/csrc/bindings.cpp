#include <torch/extension.h>
#include <vector>

namespace radar_jepa {

std::vector<torch::Tensor> radar_project_cuda(
    torch::Tensor points,
    torch::Tensor intrinsics,
    torch::Tensor extrinsics);

torch::Tensor bev_voxelize_cuda(
    torch::Tensor points,
    float x_min, float x_max,
    float y_min, float y_max,
    int grid_h, int grid_w);

torch::Tensor radar_rasterize_cuda(
    torch::Tensor coords_2d,
    torch::Tensor features,
    torch::Tensor valid,
    int height, int width, int radius);

} // namespace radar_jepa

#define CHECK_CUDA(x)       TORCH_CHECK((x).is_cuda(),       #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)      CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static std::vector<torch::Tensor> radar_project(
    torch::Tensor points,
    torch::Tensor intrinsics,
    torch::Tensor extrinsics)
{
    CHECK_INPUT(points);
    CHECK_INPUT(intrinsics);
    CHECK_INPUT(extrinsics);
    return radar_jepa::radar_project_cuda(points, intrinsics, extrinsics);
}

static torch::Tensor bev_voxelize(
    torch::Tensor points,
    float x_min, float x_max,
    float y_min, float y_max,
    int grid_h, int grid_w)
{
    CHECK_INPUT(points);
    return radar_jepa::bev_voxelize_cuda(
        points, x_min, x_max, y_min, y_max, grid_h, grid_w);
}

static torch::Tensor radar_rasterize(
    torch::Tensor coords_2d,
    torch::Tensor features,
    torch::Tensor valid,
    int height, int width, int radius)
{
    CHECK_INPUT(coords_2d);
    CHECK_INPUT(features);
    CHECK_INPUT(valid);
    return radar_jepa::radar_rasterize_cuda(
        coords_2d, features, valid, height, width, radius);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("radar_project",    &radar_project,
          "Project radar 3D points to 2D image coordinates (CUDA)");
    m.def("bev_voxelize",     &bev_voxelize,
          "Voxelize radar points into a BEV grid (CUDA)");
    m.def("radar_rasterize",  &radar_rasterize,
          "Rasterize projected radar points onto a 2D feature canvas (CUDA)");
}
