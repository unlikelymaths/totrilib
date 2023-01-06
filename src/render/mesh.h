#ifndef TOTRI_RENDER_MESH
#define TOTRI_RENDER_MESH

#include <array>
#include <torch/extension.h>

namespace totri {
namespace render {

void mesh_confocal_forward(
  const at::Tensor verts, // [B, V, 3]
  const at::Tensor faces, // [B, F, 3]
  const at::Tensor material, // [B, V, M]
  const at::Tensor scan_points, // [B, S, 3]
  const at::Tensor scan_origin, // [B, S, 3]
  at::Tensor transient, // [B, T, S]
  const float bin_width, const float bin_offset,
  const int model);

void mesh_confocal_grad_verts(
  const at::Tensor verts, // [B, V, 3]
  const at::Tensor faces, // [B, F, 3]
  const at::Tensor material, // [B, V, M]
  const at::Tensor scan_points, // [B, S, 3]
  const at::Tensor scan_origin, // [B, S, 3]
  const at::Tensor transient_grad, // [B, T, S]
  at::Tensor verts_grad, // [B, V, 3]
  at::Tensor material_grad, // [B, V, M]
  const float bin_width, const float bin_offset,
  const int model);

void mesh_exhaustive_forward(
  const at::Tensor verts, // [B, V, 3]
  const at::Tensor faces, // [B, F, 3]
  const at::Tensor material, // [B, V, M]
  const at::Tensor scan_points, // [B, S, 3]
  const at::Tensor laser_points, // [B, L, 3]
  const at::Tensor scan_origin, // [B, S, 3]
  const at::Tensor laser_origin, // [B, S, 3]
  at::Tensor transient, // [B, T, S, L]
  const float bin_width, const float bin_offset,
  const int model);

void mesh_exhaustive_grad_verts(
  const at::Tensor verts, // [B, V, 3]
  const at::Tensor faces, // [B, F, 3]
  const at::Tensor material, // [B, V, M]
  const at::Tensor scan_points, // [B, S, 3]
  const at::Tensor laser_points, // [B, L, 3]
  const at::Tensor scan_origin, // [B, S, 3]
  const at::Tensor laser_origin, // [B, S, 3]
  const at::Tensor transient_grad, // [B, T, S, L]
  at::Tensor verts_grad, // [B, V, 3]
  at::Tensor material_grad, // [B, V, M]
  const float bin_width, const float bin_offset,
  const int model);

} // namespace render
} // namespace totri

#endif // TOTRI_RENDER_MESH