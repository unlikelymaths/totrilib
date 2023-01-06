/**
 * @file
 * @copydoc render/program/mesh_confocal_forward.h
 */
#include "render/program/mesh_confocal_forward.h"

#include <optix_device.h>

#include "common/const.h"
#include "common/math.h"
#include "render/program/mesh_common.h"

namespace totri {
namespace render {
 
extern "C" __constant__ MeshConfocalForwardLaunchParams mesh_confocal_forward_launch_params;

extern "C" __global__ void __raygen__rg() {
  MeshConfocalForwardLaunchParams& p = mesh_confocal_forward_launch_params;
  const uint3 idx = optixGetLaunchIndex();
  const uint f = idx.x; // face index
  const uint s = idx.y; // scan index
  const float3 scan_point = make_float3(
    p.scan_points[p.b][s][0],
    p.scan_points[p.b][s][1],
    p.scan_points[p.b][s][2]);
  const uint3 vertex_indices = p.faces[f];
  float3 vertex_0 = make_float3(
    p.verts[p.b][vertex_indices.x][0],
    p.verts[p.b][vertex_indices.x][1],
    p.verts[p.b][vertex_indices.x][2]);
  float3 vertex_1 = make_float3(
    p.verts[p.b][vertex_indices.y][0],
    p.verts[p.b][vertex_indices.y][1],
    p.verts[p.b][vertex_indices.y][2]);
  float3 vertex_2 = make_float3(
    p.verts[p.b][vertex_indices.z][0],
    p.verts[p.b][vertex_indices.z][1],
    p.verts[p.b][vertex_indices.z][2]);
  const float3 centroid = 1.f / 3.f * (vertex_0 + vertex_1 + vertex_2);

  // Occlusion test (sensor - centroid)
  float3 ray_direction = centroid - scan_point;
  float centroid_distance = length(ray_direction);
  if (centroid_distance < 1.e-6) {
    return;
  }
  ray_direction = ray_direction / centroid_distance;
  unsigned int hit_index;
  optixTrace(
    p.handle, // handle
    scan_point, // rayOrigin
    ray_direction, // rayDirection
    0.0f, // tmin
    centroid_distance * 1.01f + 1.e-3, // tmax
    0.0f, // rayTime
    OptixVisibilityMask(255), // visibilityMask
    OPTIX_RAY_FLAG_NONE, // rayFlags
    0, // SBToffset
    0, // SBTstride
    0, // missSBTIndex
    hit_index); // p0
  if (hit_index != f) {
    return;
  }

  // Compute Origin distance
  float origin_distance = 0.f;
  if (p.scan_origin.size(1) > 0) {
    origin_distance = 2.f * length(scan_point - make_float3(
      p.scan_origin[p.b][s][0],
      p.scan_origin[p.b][s][1],
      p.scan_origin[p.b][s][2]
    ));
  }

  // Sample material properties
  float4 material = make_float4(1.f, 0.f, 0.f, 0.f);
  for (int i=0; i<p.material.size(2); ++i) {
    reinterpret_cast<float*>(&material)[i] = (1.f / 3.f) * (
      p.material[p.b][vertex_indices.x][i] +
      p.material[p.b][vertex_indices.y][i] +
      p.material[p.b][vertex_indices.z][i]);
  }

  // Sort vertices by distance/time
  float distance_0 = length(vertex_0 - scan_point);
  float distance_1 = length(vertex_1 - scan_point);
  float distance_2 = length(vertex_2 - scan_point);
  if (distance_0 > distance_1) {
    swap(distance_0, distance_1);
    swap(vertex_0, vertex_1);
  }
  if (distance_1 > distance_2) {
    swap(distance_1, distance_2);
    swap(vertex_1, vertex_2);
  }
  if (distance_0 > distance_1) {
    swap(distance_0, distance_1);
    swap(vertex_0, vertex_1);
  }

  // Compute intensity
  const float3 scan_normal = make_float3(0, 0, 1);
  const float alpha = AlphaConfocal(
    vertex_0, vertex_1, vertex_2, centroid,
    material, p.model,
    scan_point, scan_normal,
    false);
  if (alpha <= 0) {
    return;
  }

  // Draw triangles
  const float cbin_0 = (2.f * distance_0 + origin_distance - p.bin_offset) / p.bin_width;
  const float cbin_1 = (2.f * distance_1 + origin_distance - p.bin_offset) / p.bin_width;
  const float cbin_2 = (2.f * distance_2 + origin_distance - p.bin_offset) / p.bin_width;
  const int bin_0 = cbin_0;
  const int bin_2 = cbin_2;
  if (bin_0 >= p.transient.size(1) || bin_2 < 0) {
    return;
  }
  if (bin_0 == bin_2) {
    atomicAdd(&p.transient[p.b][bin_0][s], alpha);
  } else {
    const float alpha_ycenter = 2.0f * alpha / (cbin_2 - cbin_0);
    DrawTrapezoid(cbin_0,           0.f, cbin_1, alpha_ycenter, s, p);
    DrawTrapezoid(cbin_1, alpha_ycenter, cbin_2,           0.f, s, p);
  }
}

extern "C" __global__ void __closesthit__ch() {
  unsigned int triangle_idx = optixGetPrimitiveIndex();
  optixSetPayload_0(triangle_idx);
}
 
extern "C" __global__ void __miss__ms() {
  optixSetPayload_0(MAX_UINT);
}

} // namespace render
} // namespace totri
