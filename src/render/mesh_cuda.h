/**
* @file
* @brief  Transient Mesh Renderer
*/
#ifndef TOTRI_RENDER_MESH_CUDA_H
#define TOTRI_RENDER_MESH_CUDA_H

#include <torch/extension.h>

#include "common/optix.h"
#include "render/program/mesh_confocal_forward.h"
#include "render/program/mesh_confocal_grad_verts.h"
#include "render/program/mesh_exhaustive_forward.h"
#include "render/program/mesh_exhaustive_grad_verts.h"

namespace totri {
namespace render {

class MeshConfocalContext {
 public:
	MeshConfocalContext(int device);
  void Forward(
    const at::Tensor verts, // [B, V, 3]
    const at::Tensor faces, // [B, F, 3]
    const at::Tensor material, // [B, V, M]
    const at::Tensor scan_points, // [B, S, 3]
    const at::Tensor scan_origin, // [B, S, 3]
    at::Tensor transient, // [B, T, S]
    const float bin_width, const float bin_offset, const int model);
  void GradVerts(
    const at::Tensor verts, // [B, V, 3]
    const at::Tensor faces, // [B, F, 3]
    const at::Tensor material, // [B, V, M]
    const at::Tensor scan_points, // [B, S, 3]
    const at::Tensor scan_origin, // [B, S, 3]
    const at::Tensor transient_grad, // [B, T, S]
    at::Tensor verts_grad, // [B, V, 3]
    at::Tensor material_grad, // [B, V, M]
    const float bin_width, const float bin_offset, const int model);

 private:
	MeshConfocalContext(const MeshConfocalContext&) = delete;
	MeshConfocalContext& operator= (const MeshConfocalContext&) = delete;

	SharedOptixContext optix_context_;
  TriangleGas gas_;
  MeshConfocalForwardProgram forward_program_;
  MeshConfocalGradVertsProgram grad_verts_program_;
};

class MeshExhaustiveContext {
 public:
	MeshExhaustiveContext(int device);
  void Forward(
    const at::Tensor verts, // [B, V, 3]
    const at::Tensor faces, // [B, F, 3]
    const at::Tensor material, // [B, V, M]
    const at::Tensor scan_points, // [B, S, 3]
    const at::Tensor laser_points, // [B, L, 3]
    const at::Tensor scan_origin, // [B, S, 3]
    const at::Tensor laser_origin, // [B, L, 3]
    at::Tensor transient, // [B, T, S, L]
    const float bin_width, const float bin_offset, const int model);
  void GradVerts(
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

 private:
	MeshExhaustiveContext(const MeshExhaustiveContext&) = delete;
	MeshExhaustiveContext& operator= (const MeshExhaustiveContext&) = delete;

	SharedOptixContext optix_context_;
  TriangleGas gas_;
  MeshExhaustiveForwardProgram forward_program_;
  MeshExhaustiveGradVertsProgram grad_verts_program_;
};

} // namespace render
} // namespace totri

#endif // TOTRI_RENDER_MESH_CUDA_H
