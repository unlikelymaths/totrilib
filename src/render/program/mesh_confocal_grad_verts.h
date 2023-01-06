/**
 * @file
 * @brief Transient Confocal Mesh Renderer Program Verts Gradient
 */
#ifndef TOTRI_RENDER_PROGRAM_MESH_CONFOCAL_GRAD_VERTS_H
#define TOTRI_RENDER_PROGRAM_MESH_CONFOCAL_GRAD_VERTS_H

#include <optix.h>
#include <optix_stubs.h>
#include <torch/extension.h>

#include "common/buffer.h"
#include "common/optix.h"
#include "render/program/mesh_program.h"

namespace totri {
namespace render {

struct MeshConfocalGradVertsLaunchParams {
  OptixTraversableHandle handle;
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> verts; // [B, V, 3]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> material; // [B, V, M]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> scan_points; // [B, S, 3]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> scan_origin; // [B, S, 3]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> transient_grad; // [B, T, S]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> verts_grad; // [B, V, 3]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> material_grad; // [B, V, M]
  uint3* faces; // [F,]
  float bin_width;
  float bin_offset;
  int model;
  int b;
};

class MeshConfocalGradVertsProgram : public MeshProgramBase {
 public:
  MeshConfocalGradVertsProgram(int device);
	HostDeviceBuffer<MeshConfocalGradVertsLaunchParams>& launch_params();

 private:
  void Init();
	HostDeviceBuffer<MeshConfocalGradVertsLaunchParams> launch_params_;
	HostDeviceBuffer<EmptyRaygenRecord> raygen_record_;
	HostDeviceBuffer<EmptyMissRecord> miss_record_;
	HostDeviceBuffer<EmptyHitgroupRecord> hitgroup_record_;
};

} // namespace render
} // namespace totri

#endif // TOTRI_RENDER_PROGRAM_MESH_CONFOCAL_GRAD_VERTS_H
