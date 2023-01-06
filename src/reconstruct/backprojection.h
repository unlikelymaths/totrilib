#ifndef TOTRI_RECONSTRUCT_BACKPROJECTION
#define TOTRI_RECONSTRUCT_BACKPROJECTION

#include <array>
#include <torch/extension.h>

namespace totri {
namespace reconstruct {

void backprojection_exhaustive_arellano_forward(
  const at::Tensor transient, // [B, N, S, L]
  const at::Tensor scan_points, // [B, 3, S]
  const at::Tensor laser_points, // [B, 3, L]
  const at::Tensor scan_origin, // [B, 3, S]
  const at::Tensor laser_origin, // [B, 3, L]
  at::Tensor volume, // [B, D, H, W]
  const std::array<float, 3>& volume_start,
  const std::array<float, 3>& volume_end,
  const float bin_width, const float bin_offset);

void backprojection_confocal_velten_forward(
  const at::Tensor transient, // [B, T, S]
  const at::Tensor scan_points, // [B, S, 3]
  const at::Tensor scan_origin, // [B, S, 3]
  at::Tensor volume, // [B, D, H, W]
  const std::array<float, 3>& volume_start,
  const std::array<float, 3>& volume_end,
  const float bin_width, const float bin_offset);

void backprojection_exhaustive_velten_forward(
  const at::Tensor transient, // [B, N, S, L]
  const at::Tensor scan_points,  // [B, S, 3]
  const at::Tensor laser_points, // [B, L, 3]
  const at::Tensor scan_origin,  // [B, S, 3]
  const at::Tensor laser_origin, // [B, L, 3]
  at::Tensor volume, // [B, D, H, W]
  const std::array<float, 3>& volume_start,
  const std::array<float, 3>& volume_end,
  const float bin_width, const float bin_offset);

} // namespace reconstruct
} // namespace totri

#endif // TOTRI_RECONSTRUCT_BACKPROJECTION
