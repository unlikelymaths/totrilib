/**
 * @file
 * @copydoc render/program/mesh_exhaustive_grad_verts.h
 */
#include "render/program/mesh_exhaustive_grad_verts.h"

#include <optix_device.h>

#include "common/const.h"
#include "common/math.h"
#include "render/program/mesh_common.h"

namespace totri {
namespace render {
 
extern "C" __constant__ MeshExhaustiveGradVertsLaunchParams mesh_exhaustive_grad_verts_launch_params;

extern "C" __global__ void __raygen__rg() {
  MeshExhaustiveGradVertsLaunchParams& p = mesh_exhaustive_grad_verts_launch_params;
  const uint3 idx = optixGetLaunchIndex();
  const uint f = idx.x; // face index
  const uint s = idx.y; // scan index
  const uint l = idx.z; // laser index
  const float3 scan_point = make_float3(
    p.scan_points[p.b][s][0],
    p.scan_points[p.b][s][1],
    p.scan_points[p.b][s][2]);
  const float3 laser_point = make_float3(
    p.laser_points[p.b][l][0],
    p.laser_points[p.b][l][1],
    p.laser_points[p.b][l][2]);
  uint3 vertex_indices = p.faces[f];
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

  // Occlusion test (laser - centroid)
  float3 ray_direction = centroid - laser_point;
  float centroid_distance = length(ray_direction);
  if (centroid_distance < 1.e-6) {
    return;
  }
  ray_direction = ray_direction / centroid_distance;
  unsigned int hit_index;
  optixTrace(
    p.handle, // handle
    laser_point, // rayOrigin
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

  // Occlusion test (sensor - centroid)
  ray_direction = centroid - scan_point;
  centroid_distance = length(ray_direction);
  if (centroid_distance < 1.e-6) {
    return;
  }
  ray_direction = ray_direction / centroid_distance;
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
    origin_distance += length(scan_point - make_float3(
      p.scan_origin[p.b][s][0],
      p.scan_origin[p.b][s][1],
      p.scan_origin[p.b][s][2]
    ));
  }
  if (p.laser_origin.size(1) > 0) {
    origin_distance += length(laser_point - make_float3(
      p.laser_origin[p.b][l][0],
      p.laser_origin[p.b][l][1],
      p.laser_origin[p.b][l][2]
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
  float scan_distance_0 = length(vertex_0 - scan_point);
  float scan_distance_1 = length(vertex_1 - scan_point);
  float scan_distance_2 = length(vertex_2 - scan_point);
  float laser_distance_0 = length(vertex_0 - laser_point);
  float laser_distance_1 = length(vertex_1 - laser_point);
  float laser_distance_2 = length(vertex_2 - laser_point);
  float distance_0 = scan_distance_0 + laser_distance_0;
  float distance_1 = scan_distance_1 + laser_distance_1;
  float distance_2 = scan_distance_2 + laser_distance_2;
  if (distance_0 > distance_1) {
    swap(scan_distance_0, scan_distance_1);
    swap(laser_distance_0, laser_distance_1);
    swap(distance_0, distance_1);
    swap(vertex_0, vertex_1);
    swap(vertex_indices.x, vertex_indices.y);
  }
  if (distance_1 > distance_2) {
    swap(scan_distance_1, scan_distance_2);
    swap(laser_distance_1, laser_distance_2);
    swap(distance_1, distance_2);
    swap(vertex_1, vertex_2);
    swap(vertex_indices.y, vertex_indices.z);
  }
  if (distance_0 > distance_1) {
    swap(scan_distance_0, scan_distance_1);
    swap(laser_distance_0, laser_distance_1);
    swap(distance_0, distance_1);
    swap(vertex_0, vertex_1);
    swap(vertex_indices.x, vertex_indices.y);
  }

  // Compute intensity
  const float3 scan_normal = make_float3(0, 0, 1);
  const float3 laser_normal = make_float3(0, 0, 1);
  float alpha = AlphaExhaustive(
    vertex_0, vertex_1, vertex_2, centroid,
    material, p.model, 
    scan_point, scan_normal,
    laser_point, laser_normal,
    false);
  if (alpha <= 0 || isnan(alpha)) {
    return;
  }
  const vert3 dalpha_dtriangle = AlphaExhaustiveGradVerts(
    vertex_0, vertex_1, vertex_2, centroid,
    material, p.model,
    scan_point, scan_normal,
    laser_point, laser_normal,
    false, alpha);
  const float4 dalpha_dmaterial = AlphaExhaustiveGradMaterial(
    vertex_0, vertex_1, vertex_2, centroid,
    material, p.model,
    scan_point, scan_normal,
    laser_point, laser_normal,
    false, alpha);

  // Sample from triangles
  vert3 dloss_dtriangle;
  float4 dloss_dmaterial;
  const float cbin_0 = (distance_0 + origin_distance - p.bin_offset) / p.bin_width;
  const float cbin_1 = (distance_1 + origin_distance - p.bin_offset) / p.bin_width;
  const float cbin_2 = (distance_2 + origin_distance - p.bin_offset) / p.bin_width;
  const int bin_0 = cbin_0;
  const int bin_1 = cbin_1;
  const int bin_2 = cbin_2;
  if (bin_0 >= p.transient_grad.size(1) || bin_2 < 0) {
    return;
  }
  if (bin_0 == bin_2) {
    float dloss_dtransient = p.transient_grad[p.b][bin_0][s][l];
    dloss_dtriangle = dloss_dtransient * dalpha_dtriangle;
    dloss_dmaterial = dloss_dtransient * dalpha_dmaterial;
  } else {
    float dloss_dtriangle_part0 = 0;
    vert3 dloss_dtriangle_part1;
    const float ycenter = 2.0f / (cbin_2 - cbin_0);
    const vert3 dycenter_dtriangle = (-2.f / (p.bin_width * powf(cbin_2 - cbin_0, 2.f))) * vert3(
      (vertex_0 - laser_point) / (-laser_distance_0) + (vertex_0 - scan_point) / (-scan_distance_0),
      make_float3(0.f, 0.f, 0.f),
      (vertex_2 - laser_point) / ( laser_distance_2) + (vertex_2 - scan_point) / ( scan_distance_2)
    );
    const vert3 dcbin_0_dtriangle = vert3(
      (1.f / (p.bin_width * laser_distance_0)) * (vertex_0 - laser_point) + (1.f / (p.bin_width * scan_distance_0)) * (vertex_0 - scan_point),
      make_float3(0.f, 0.f, 0.f),
      make_float3(0.f, 0.f, 0.f));
    const vert3 dcbin_1_dtriangle = vert3(
      make_float3(0.f, 0.f, 0.f),
      (1.f / (p.bin_width * laser_distance_1)) * (vertex_1 - laser_point) + (1.f / (p.bin_width * scan_distance_1)) * (vertex_1 - scan_point),
      make_float3(0.f, 0.f, 0.f));
    const vert3 dcbin_2_dtriangle = vert3(
      make_float3(0.f, 0.f, 0.f),
      make_float3(0.f, 0.f, 0.f),
      (1.f / (p.bin_width * laser_distance_2)) * (vertex_2 - laser_point) + (1.f / (p.bin_width * scan_distance_2)) * (vertex_2 - scan_point));
    DrawTrapezoidGrad(
      dloss_dtriangle_part0, dloss_dtriangle_part1,
      cbin_0,    0.f, cbin_1, ycenter,
      dcbin_0_dtriangle, dcbin_1_dtriangle,
      vert3(), dycenter_dtriangle,
      s, l, p);
    DrawTrapezoidGrad(
      dloss_dtriangle_part0, dloss_dtriangle_part1,
      cbin_1, ycenter, cbin_2,    0.f,
      dcbin_1_dtriangle, dcbin_2_dtriangle,
      dycenter_dtriangle, vert3(),
      s, l, p);
    dloss_dtriangle = dalpha_dtriangle * dloss_dtriangle_part0 + alpha * dloss_dtriangle_part1;
    dloss_dmaterial = dalpha_dmaterial * dloss_dtriangle_part0;
  }

  if (p.verts_grad.size(2) > 0) {
    atomicAdd(&p.verts_grad[p.b][vertex_indices.x][0], dloss_dtriangle.vert0.x);
    atomicAdd(&p.verts_grad[p.b][vertex_indices.x][1], dloss_dtriangle.vert0.y);
    atomicAdd(&p.verts_grad[p.b][vertex_indices.x][2], dloss_dtriangle.vert0.z);
    atomicAdd(&p.verts_grad[p.b][vertex_indices.y][0], dloss_dtriangle.vert1.x);
    atomicAdd(&p.verts_grad[p.b][vertex_indices.y][1], dloss_dtriangle.vert1.y);
    atomicAdd(&p.verts_grad[p.b][vertex_indices.y][2], dloss_dtriangle.vert1.z);
    atomicAdd(&p.verts_grad[p.b][vertex_indices.z][0], dloss_dtriangle.vert2.x);
    atomicAdd(&p.verts_grad[p.b][vertex_indices.z][1], dloss_dtriangle.vert2.y);
    atomicAdd(&p.verts_grad[p.b][vertex_indices.z][2], dloss_dtriangle.vert2.z);
  }
  for (int i=0; i<p.material_grad.size(2); ++i) {
    const float grad = (1.f / 3.f) * reinterpret_cast<float*>(&dloss_dmaterial)[i];
    atomicAdd(&p.material_grad[p.b][vertex_indices.x][i], grad);
    atomicAdd(&p.material_grad[p.b][vertex_indices.y][i], grad);
    atomicAdd(&p.material_grad[p.b][vertex_indices.z][i], grad);
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
