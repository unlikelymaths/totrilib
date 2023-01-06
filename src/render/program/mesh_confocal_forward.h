/**
 * @file
 * @brief Transient Confocal Mesh Renderer Program Forward
 */
#ifndef TOTRI_RENDER_PROGRAM_MESH_CONFOCAL_FORWARD_H
#define TOTRI_RENDER_PROGRAM_MESH_CONFOCAL_FORWARD_H

#include <optix.h>
#include <optix_stubs.h>
#include <torch/extension.h>

#include "common/buffer.h"
#include "common/optix.h"
#include "render/program/mesh_program.h"

namespace totri {
namespace render {

struct MeshConfocalForwardLaunchParams {
  OptixTraversableHandle handle;
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> verts; // [B, V, 3]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> material; // [B, V, M]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> scan_points; // [B, S, 3]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> scan_origin; // [B, S, 3]
  at::PackedTensorAccessor32<float, 3, at::DefaultPtrTraits> transient; // [B, T, S]
  uint3* faces; // [F,]
  float bin_width;
  float bin_offset;
  int model;
  int b;
};

class MeshConfocalForwardProgram : public MeshProgramBase {
 public:
  MeshConfocalForwardProgram(int device);
	HostDeviceBuffer<MeshConfocalForwardLaunchParams>& launch_params();

 private:
  void Init();
	HostDeviceBuffer<MeshConfocalForwardLaunchParams> launch_params_;
	HostDeviceBuffer<EmptyRaygenRecord> raygen_record_;
	HostDeviceBuffer<EmptyMissRecord> miss_record_;
	HostDeviceBuffer<EmptyHitgroupRecord> hitgroup_record_;
};

} // namespace render
} // namespace totri

#endif // TOTRI_RENDER_PROGRAM_MESH_CONFOCAL_FORWARD_H
