#include "render/mesh.h"

#include "common/context_manager.h"
#include "render/mesh_cuda.h"

namespace totri {
namespace render {

void mesh_confocal_forward(
    const at::Tensor verts, // [B, V, 3]
    const at::Tensor faces, // [B, F, 3]
    const at::Tensor material, // [B, V, M]
    const at::Tensor scan_points, // [B, S, 3]
    const at::Tensor scan_origin, // [B, S, 3]
    at::Tensor transient, // [B, T, S]
    const float bin_width, const float bin_offset, const int model) {
  MeshConfocalContext* context = ContextManager<MeshConfocalContext>::Get(0);
  context->Forward(verts, faces, material, scan_points, scan_origin,
                   transient, bin_width, bin_offset, model);
}

void mesh_confocal_grad_verts(
    const at::Tensor verts, // [B, V, 3]
    const at::Tensor faces, // [B, F, 3]
    const at::Tensor material, // [B, V, M]
    const at::Tensor scan_points, // [B, S, 3]
    const at::Tensor scan_origin, // [B, S, 3]
    const at::Tensor transient_grad, // [B, T, S]
    at::Tensor verts_grad, // [B, V, 3]
    at::Tensor material_grad, // [B, V, M]
    const float bin_width, const float bin_offset, const int model) {
  MeshConfocalContext* context = ContextManager<MeshConfocalContext>::Get(0);
  context->GradVerts(verts, faces, material, scan_points, scan_origin,
                     transient_grad, verts_grad, material_grad,
                     bin_width, bin_offset, model);
}

void mesh_exhaustive_forward(
    const at::Tensor verts, // [B, V, 3]
    const at::Tensor faces, // [B, F, 3]
    const at::Tensor material, // [B, V, M]
    const at::Tensor scan_points, // [B, S, 3]
    const at::Tensor laser_points, // [B, L, 3]
    const at::Tensor scan_origin, // [B, S, 3]
    const at::Tensor laser_origin, // [B, L, 3]
    at::Tensor transient, // [B, T, S, L]
    const float bin_width, const float bin_offset, const int model) {
  MeshExhaustiveContext* context = ContextManager<MeshExhaustiveContext>::Get(0);
  context->Forward(verts, faces, material, scan_points, laser_points,
                   scan_origin, laser_origin, transient,
                   bin_width, bin_offset, model);
}

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
    const int model) {
  MeshExhaustiveContext* context = ContextManager<MeshExhaustiveContext>::Get(0);
  context->GradVerts(verts, faces, material, scan_points, laser_points,
                     scan_origin, laser_origin, transient_grad, verts_grad,
                     material_grad, bin_width, bin_offset, model);
}


} // namespace render
} // namespace totri
