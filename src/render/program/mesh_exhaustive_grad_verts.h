/**
 * @file
 * @brief Transient Exhaustive Mesh Renderer Program Verts Gradient
 */
#ifndef TOTRI_RENDER_PROGRAM_MESH_EXHAUSTIVE_GRAD_VERTS_H
#define TOTRI_RENDER_PROGRAM_MESH_EXHAUSTIVE_GRAD_VERTS_H

#include <optix.h>
#include <optix_stubs.h>
#include <torch/extension.h>

#include "common/buffer.h"
#include "common/optix.h"
#include "render/program/mesh_program.h"

namespace totri {
namespace render {

struct MeshExhaustiveGradVertsLaunchParams {
  OptixTraversableHandle handle;
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> verts; // [B, V, 3]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> material; // [B, V, M]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> scan_points; // [B, S, 3]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> laser_points; // [B, L, 3]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> scan_origin; // [B, S, 3]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> laser_origin; // [B, L, 3]
  at::PackedTensorAccessor32<float, 4, at::DefaultPtrTraits> transient_grad; // [B, T, S, L]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> verts_grad; // [B, V, 3]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> material_grad; // [B, V, M]
  uint3* faces; // [F,]
  float bin_width;
  float bin_offset;
  int model;
  int b;
};

class MeshExhaustiveGradVertsProgram : public MeshProgramBase {
 public:
  MeshExhaustiveGradVertsProgram(int device);
	HostDeviceBuffer<MeshExhaustiveGradVertsLaunchParams>& launch_params();

 private:
  void Init();
	HostDeviceBuffer<MeshExhaustiveGradVertsLaunchParams> launch_params_;
	HostDeviceBuffer<EmptyRaygenRecord> raygen_record_;
	HostDeviceBuffer<EmptyMissRecord> miss_record_;
	HostDeviceBuffer<EmptyHitgroupRecord> hitgroup_record_;
};

} // namespace render
} // namespace totri

#endif // TOTRI_RENDER_PROGRAM_MESH_EXHAUSTIVE_GRAD_VERTS_H
