#ifndef TOTRI_RENDER_VOLUME_CUDA
#define TOTRI_RENDER_VOLUME_CUDA

#include <array>
#include <torch/extension.h>

namespace totri {
namespace render {

void volume_confocal_forward_cuda(
  const at::Tensor volume, // X x Y x Z
  const at::Tensor measurement_points, // 3 x N
  at::Tensor transient, // N x T
  const std::array<float, 3>& volume_origin, const std::array<float, 3>& volume_size,
  const float bin_width, const float bin_offset
);

void volume_confocal_grad_volume_cuda(
  at::Tensor volume, // X x Y x Z
  at::Tensor measurement_points, // 3 x N
  at::Tensor transient, // N x T
  const std::array<float, 3>& volume_origin, const std::array<float, 3>& volume_size,
  float bin_width, float bin_offset
);

void volume_confocal_grad_measurement_points_cuda(
  at::Tensor measurement_points_grad, // 3 x N
  const at::Tensor volume, // X x Y x Z
  const at::Tensor measurement_points, // 3 x N
  const at::Tensor transient_grad, // N x T
  const std::array<float, 3>& volume_origin, const std::array<float, 3>& volume_size,
  float bin_width, float bin_offset
);

void volume_render_exhaustive_forward_cuda(
  const at::Tensor volume, // [B, D, H, W]
  const at::Tensor scan_points, // [B, 3, S]
  const at::Tensor laser_points, // [B, 3, L]
  const at::Tensor scan_origin, // [B, 3, S]
  const at::Tensor laser_origin, // [B, 3, L]
  at::Tensor transient, // [B, N, S, L]
  const std::array<float, 3>& volume_start, const std::array<float, 3>& volume_end,
  const float bin_width, const float bin_offset
);

void volume_render_exhaustive_grad_volume_cuda(
  const at::Tensor transient_grad, // [B, N, S, L]
  const at::Tensor scan_points, // [B, 3, S]
  const at::Tensor laser_points, // [B, 3, L]
  const at::Tensor scan_origin, // [B, 3, S]
  const at::Tensor laser_origin, // [B, 3, L]
  at::Tensor volume_grad, // [B, D, H, W]
  const std::array<float, 3>& volume_start, const std::array<float, 3>& volume_end,
  const float bin_width, const float bin_offset
);

} // namespace render
} // namespace totri

#endif // TOTRI_RENDER_VOLUME_CUDA
