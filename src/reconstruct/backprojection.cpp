#include "reconstruct/backprojection.h"
#include "reconstruct/backprojection_cuda.h"

namespace totri {
namespace reconstruct {

void backprojection_exhaustive_arellano_forward(
    const at::Tensor transient, // [B, N, S, L]
    const at::Tensor scan_points, // [B, 3, S]
    const at::Tensor laser_points, // [B, 3, L]
    const at::Tensor scan_origin, // [B, 3, S]
    const at::Tensor laser_origin, // [B, 3, L]
    at::Tensor volume, // [B, D, H, W]
    const std::array<float, 3>& volume_start, const std::array<float, 3>& volume_end,
    const float bin_width, const float bin_offset) {
  backprojection_exhaustive_arellano_forward_cuda(
    transient, scan_points, laser_points, scan_origin, laser_origin, volume,
    volume_start, volume_end,
    bin_width, bin_offset);
}

void backprojection_confocal_velten_forward(
    const at::Tensor transient, // [B, T, S]
    const at::Tensor scan_points, // [B, S, 3]
    const at::Tensor scan_origin, // [B, S, 3]
    at::Tensor volume, // [B, D, H, W]
    const std::array<float, 3>& volume_start,
    const std::array<float, 3>& volume_end,
    const float bin_width, const float bin_offset) {
  backprojection_confocal_velten_forward_cuda(
    transient, scan_points, scan_origin, volume,
    volume_start, volume_end,
    bin_width, bin_offset);
}

void backprojection_exhaustive_velten_forward(
    const at::Tensor transient, // [B, N, S, L]
    const at::Tensor scan_points,  // [B, S, 3]
    const at::Tensor laser_points, // [B, L, 3]
    const at::Tensor scan_origin,  // [B, S, 3]
    const at::Tensor laser_origin, // [B, L, 3]
    at::Tensor volume, // [B, D, H, W]
    const std::array<float, 3>& volume_start, const std::array<float, 3>& volume_end,
    const float bin_width, const float bin_offset) {
  backprojection_exhaustive_velten_forward_cuda(
    transient, scan_points, laser_points, scan_origin, laser_origin, volume,
    volume_start, volume_end,
    bin_width, bin_offset);
}

} // namespace reconstruct
} // namespace totri
