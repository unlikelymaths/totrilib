#include "render/volume.h"
#include "render/volume_cuda.h"

namespace totri {
namespace render {

void volume_confocal_forward(
    const at::Tensor volume, // X x Y x Z
    const at::Tensor measurement_points, // 3 x N
    at::Tensor transient, // N x T
    const std::array<float, 3> volume_origin, const std::array<float, 3> volume_size,
    const float bin_width, const float bin_offset) {
  // Check sizes
  if (measurement_points.size(1) != transient.size(0)) {
    throw std::range_error("Measurement and transient dimensions do not match.");
  }
  // Apply RenderOperator
  volume_confocal_forward_cuda(
    volume, measurement_points, transient,
    volume_origin, volume_size,
    bin_width, bin_offset);
  return;
}

void volume_confocal_grad_volume(
    at::Tensor volume, // X x Y x Z
    at::Tensor measurement_points, // 3 x N
    at::Tensor transient, // N x T
    std::array<float, 3> volume_origin, std::array<float, 3> volume_size,
    float bin_width, float bin_offset) {
  // Check sizes
  if (measurement_points.size(1) != transient.size(0)) {
    throw std::range_error("Measurement and transient dimensions do not match.");
  }
  // Apply RenderOperator Transposed
  volume_confocal_grad_volume_cuda(
    volume, measurement_points, transient,
    volume_origin, volume_size,
    bin_width, bin_offset);
  return;
}

void volume_confocal_grad_measurement_points(
    at::Tensor measurement_points_grad, // 3 x N
    const at::Tensor volume, // X x Y x Z
    const at::Tensor measurement_points, // 3 x N
    const at::Tensor transient_grad, // N x T
    std::array<float, 3> volume_origin, std::array<float, 3> volume_size,
    float bin_width, float bin_offset) {
  // Check sizes
  if (measurement_points_grad.dim() != 2) {
    throw std::range_error("Gradient must be 2D.");
  }
  if (volume.dim() != 3) {
    throw std::range_error("volume must be 3D.");
  }
  if (measurement_points.dim() != 2) {
    throw std::range_error("Measurement points must be 2D.");
  }
  if (transient_grad.dim() != 2) {
    throw std::range_error("Transient must be 2D.");
  }
  if (measurement_points.size(0) != 3) {
    throw std::range_error("Measurement points must have shape 3xN.");
  }
  if (measurement_points_grad.size(0) != 3) {
    throw std::range_error("Gradient must have shape 3xN.");
  }
  if (measurement_points.size(1) != transient_grad.size(0)) {
    throw std::range_error("Measurement points and transient dimensions do not match.");
  }
  if (measurement_points_grad.size(1) != measurement_points.size(1)) {
    throw std::range_error("Measurement points and gradient dimensions do not match.");
  }
  // Apply RenderOperator Gradient
  volume_confocal_grad_measurement_points_cuda(
    measurement_points_grad, volume, measurement_points, transient_grad,
    volume_origin, volume_size,
    bin_width, bin_offset);
  return;
}

void volume_render_exhaustive_forward(
    const at::Tensor volume, // [B, D, H, W]
    const at::Tensor scan_points, // [B, 3, S]
    const at::Tensor laser_points, // [B, 3, L]
    const at::Tensor scan_origin, // [B, 3, S]
    const at::Tensor laser_origin, // [B, 3, L]
    at::Tensor transient, // [B, N, S, L]
    const std::array<float, 3>& volume_start, const std::array<float, 3>& volume_end,
    const float bin_width, const float bin_offset) {
  volume_render_exhaustive_forward_cuda(
    volume, scan_points, laser_points, scan_origin, laser_origin, transient,
    volume_start, volume_end,
    bin_width, bin_offset);
}

void volume_render_exhaustive_grad_volume(
    const at::Tensor transient_grad, // [B, N, S, L]
    const at::Tensor scan_points, // [B, 3, S]
    const at::Tensor laser_points, // [B, 3, L]
    const at::Tensor scan_origin, // [B, 3, S]
    const at::Tensor laser_origin, // [B, 3, L]
    at::Tensor volume_grad, // [B, D, H, W]
    const std::array<float, 3>& volume_start, const std::array<float, 3>& volume_end,
    const float bin_width, const float bin_offset) {
  volume_render_exhaustive_grad_volume_cuda(
    transient_grad, scan_points, laser_points, scan_origin, laser_origin, volume_grad,
    volume_start, volume_end,
    bin_width, bin_offset);
}

} // namespace render
} // namespace totri
