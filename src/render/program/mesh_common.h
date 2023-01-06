/**
 * @file
 * @brief Transient Mesh Renderer Program Functions
 */
#ifndef TOTRI_RENDER_PROGRAM_MESH_COMMON_H
#define TOTRI_RENDER_PROGRAM_MESH_COMMON_H

#include "common/const.h"
#include "common/math.h"
#include "common/triangle.h"
#include "render/program/mesh_confocal_forward.h"
#include "render/program/mesh_confocal_grad_verts.h"
#include "render/program/mesh_exhaustive_forward.h"
#include "render/program/mesh_exhaustive_grad_verts.h"

namespace totri {
namespace render {

/**
 * @brief Evaluate value of the line at point x
 * 
 * @param x   Point where the line is evaluated
 * @param x_0 First x coordinate of the line
 * @param y_0 First y coordinate of the line
 * @param x_1 Second x coordinate of the line
 * @param y_1 Second y coordinate of the line
 */
__device__ inline float LinearInterp(
    const float x,
    const float x_0, const float y_0,
    const float x_1, const float y_1) {
  return y_0 + (x - x_0) / (x_1 - x_0) * (y_1 - y_0);
}

/**
 * @brief Compute area under the line between b_0 and b_1
 * 
 * @param b_0 Start of the interval
 * @param b_1 End of the interval
 * @param x_0 First x coordinate of the line
 * @param y_0 First y coordinate of the line
 * @param x_1 Second x coordinate of the line
 * @param y_1 Second y coordinate of the line
 */
__device__ inline float AreaTrapezoid(
    const float b_0, const float b_1,
    const float x_0, const float y_0,
    const float x_1, const float y_1) {
  return (b_1 - b_0) * LinearInterp(
    0.5f * (b_0 + b_1), x_0, y_0, x_1, y_1);
}

/**
 * @brief Add trapezoid to transient image
 * 
 * @param cbin_0  Start point of the trapezoid
 * @param alpha_0 Start value of the trapezoid
 * @param cbin_1  End point of the trapezoid
 * @param alpha_1 End value of the trapezoid
 */
__device__ inline void DrawTrapezoid(
    float cbin_0, float alpha_y0,
    float cbin_1, float alpha_y1,
    int s, MeshConfocalForwardLaunchParams& p) {
  const int bin_0 = cbin_0;
  const int bin_1 = cbin_1;
  if (bin_0 == bin_1) {
    if (bin_0 >= 0 && bin_0 < p.transient.size(1)) { 
      atomicAdd(&p.transient[p.b][bin_0][s],
                0.5f * (cbin_1 - cbin_0) * (alpha_y0 + alpha_y1));
    }
  } else {
    if (bin_0 >= 0 && bin_0 < p.transient.size(1)) {
      atomicAdd(&p.transient[p.b][bin_0][s], AreaTrapezoid(
        cbin_0, bin_0 + 1,
        cbin_0, alpha_y0,
        cbin_1, alpha_y1));
    }
    for (int i = bin_0 + 1; i < bin_1; ++i) {
      if (i >= 0 && i < p.transient.size(1)) {
        atomicAdd(&p.transient[p.b][i][s], AreaTrapezoid(
          i, i + 1,
          cbin_0, alpha_y0,
          cbin_1, alpha_y1));
      }
    }
    if (bin_1 >= 0 && bin_1 < p.transient.size(1)) {
      atomicAdd(&p.transient[p.b][bin_1][s], AreaTrapezoid(
        bin_1, cbin_1,
        cbin_0, alpha_y0,
        cbin_1, alpha_y1));
    }
  }
}

/**
 * @brief Compute gradient of add trapezoid to transient image
 * 
 * @param cbin_0  Start point of the trapezoid
 * @param alpha_0 Start value of the trapezoid
 * @param time_1  End point of the trapezoid
 * @param alpha_1 End value of the trapezoid
 */
__device__ inline void DrawTrapezoidGrad(
    float& dloss_dtriangle_part0,
    vert3& dloss_dtriangle_part1,
    float cbin_0, float y_0,
    float cbin_1, float y_1,
    const vert3& dcbin_0_dtriangle, const vert3& dcbin_1_dtriangle,
    const vert3& dy0_dtriangle, const vert3& dy1_dtriangle,
    int s, MeshConfocalGradVertsLaunchParams& p) {
  const int bin_0 = cbin_0;
  const int bin_1 = cbin_1;
  if (bin_0 == bin_1) {
    if (bin_0 >= 0 && bin_0 < p.transient_grad.size(1)) {
      const float transient_grad = p.transient_grad[p.b][bin_0][s];
      dloss_dtriangle_part0 +=
        transient_grad * (
          0.5f * (cbin_1 - cbin_0) * (y_0 + y_1)
        );
      dloss_dtriangle_part1 +=
        transient_grad * (
          (0.5f * (y_0 + y_1)) * (dcbin_1_dtriangle - dcbin_0_dtriangle) +
          (0.5f * (cbin_1 - cbin_0)) * (dy0_dtriangle + dy1_dtriangle)
        );
    }
  } else {
    if (bin_0 >= 0 && bin_0 < p.transient_grad.size(1)) {
      const float transient_grad = p.transient_grad[p.b][bin_0][s];
      dloss_dtriangle_part0 +=
        transient_grad * AreaTrapezoid(
          cbin_0, bin_0 + 1,
          cbin_0, y_0,
          cbin_1, y_1
        );
      const float base = bin_0 + 1 - cbin_0;
      dloss_dtriangle_part1 +=
        transient_grad * (
          dy0_dtriangle * base -
          dcbin_0_dtriangle * y_0 -
          dcbin_0_dtriangle * (base * (y_1 - y_0) / (cbin_1 - cbin_0)) +
          (0.5f * base * base) *
            (
              (dy1_dtriangle - dy0_dtriangle) * (cbin_1 - cbin_0) -
              (y_1 - y_0) * (dcbin_1_dtriangle - dcbin_0_dtriangle)
            ) /
            (powf(cbin_1 - cbin_0, 2.f))
        );
    }
    for (int i = bin_0 + 1; i < bin_1; ++i) {
      if (i >= 0 && i < p.transient_grad.size(1)) {
        const float transient_grad = p.transient_grad[p.b][i][s];
        dloss_dtriangle_part0 +=
          transient_grad * AreaTrapezoid(
            i, i + 1,
            cbin_0, y_0,
            cbin_1, y_1
          );
        dloss_dtriangle_part1 +=
          transient_grad * (
            dy0_dtriangle -
            dcbin_0_dtriangle * (y_1 - y_0) / (cbin_1 - cbin_0) +
            ((i + 0.5f) - cbin_0) *
              (
                (dy1_dtriangle - dy0_dtriangle) * (cbin_1 - cbin_0) -
                (y_1 - y_0) * (dcbin_1_dtriangle - dcbin_0_dtriangle)
              ) /
              (powf(cbin_1 - cbin_0, 2.f))
          );
      }
    }
    if (bin_1 >= 0 && bin_1 < p.transient_grad.size(1)) {
      const float transient_grad = p.transient_grad[p.b][bin_1][s];
      dloss_dtriangle_part0 +=
        transient_grad * AreaTrapezoid(
          bin_1 , cbin_1,
          cbin_0, y_0,
          cbin_1, y_1
        );
      dloss_dtriangle_part1 +=
        transient_grad * (
          dcbin_1_dtriangle * (
            y_0 + (0.5f * (bin_1 + cbin_1) - cbin_0) * (y_1 - y_0) /
            (cbin_1 - cbin_0)) +
          (cbin_1 - bin_1) * (
            dy0_dtriangle +
            (0.5f * dcbin_1_dtriangle - dcbin_0_dtriangle) *
              (y_1 - y_0) / (cbin_1 - cbin_0) +
            (0.5f * (bin_1 + cbin_1) - cbin_0) *
              (
                (dy1_dtriangle - dy0_dtriangle) * (cbin_1 - cbin_0) -
                (y_1 - y_0) * (dcbin_1_dtriangle - dcbin_0_dtriangle)
              ) /
              (powf(cbin_1 - cbin_0, 2.f))
          )
        );
    }
  }
}

/**
 * @brief Add trapezoid to transient image
 * 
 * @param time_0  Start point of the trapezoid
 * @param alpha_0 Start value of the trapezoid
 * @param time_1  End point of the trapezoid
 * @param alpha_1 End value of the trapezoid
 */
__device__ inline void DrawTrapezoid(
    float cbin_0, float alpha_y0,
    float cbin_1, float alpha_y1,
    int s, int l, MeshExhaustiveForwardLaunchParams& p) {
  const int bin_0 = cbin_0;
  const int bin_1 = cbin_1;
  if (bin_0 == bin_1) {
    if (bin_0 >= 0 && bin_0 < p.transient.size(1)) { 
      atomicAdd(&p.transient[p.b][bin_0][s][l],
                0.5f * (cbin_1 - cbin_0) * (alpha_y0 + alpha_y1));
    }
  } else {
    if (bin_0 >= 0 && bin_0 < p.transient.size(1)) {
      atomicAdd(&p.transient[p.b][bin_0][s][l], AreaTrapezoid(
        cbin_0, bin_0 + 1,
        cbin_0, alpha_y0,
        cbin_1, alpha_y1));
    }
    for (int i = bin_0 + 1; i < bin_1; ++i) {
      if (i >= 0 && i < p.transient.size(1)) {
        atomicAdd(&p.transient[p.b][i][s][l], AreaTrapezoid(
          i, i + 1,
          cbin_0, alpha_y0,
          cbin_1, alpha_y1));
      }
    }
    if (bin_1 >= 0 && bin_1 < p.transient.size(1)) {
      atomicAdd(&p.transient[p.b][bin_1][s][l], AreaTrapezoid(
        bin_1, cbin_1,
        cbin_0, alpha_y0,
        cbin_1, alpha_y1));
    }
  }
}

/**
 * @brief Compute gradient of add trapezoid to transient image
 * 
 * @param cbin_0  Start point of the trapezoid
 * @param alpha_0 Start value of the trapezoid
 * @param time_1  End point of the trapezoid
 * @param alpha_1 End value of the trapezoid
 */
__device__ inline void DrawTrapezoidGrad(
    float& dloss_dtriangle_part0,
    vert3& dloss_dtriangle_part1,
    float cbin_0, float y_0,
    float cbin_1, float y_1,
    const vert3& dcbin_0_dtriangle, const vert3& dcbin_1_dtriangle,
    const vert3& dy0_dtriangle, const vert3& dy1_dtriangle,
    int s, int l, MeshExhaustiveGradVertsLaunchParams& p) {
  const int bin_0 = cbin_0;
  const int bin_1 = cbin_1;
  if (bin_0 == bin_1) {
    if (bin_0 >= 0 && bin_0 < p.transient_grad.size(1)) {
      const float transient_grad = p.transient_grad[p.b][bin_0][s][l];
      dloss_dtriangle_part0 +=
        transient_grad * (
          0.5f * (cbin_1 - cbin_0) * (y_0 + y_1)
        );
      dloss_dtriangle_part1 +=
        transient_grad * (
          (0.5f * (y_0 + y_1)) * (dcbin_1_dtriangle - dcbin_0_dtriangle) +
          (0.5f * (cbin_1 - cbin_0)) * (dy0_dtriangle + dy1_dtriangle)
        );
    }
  } else {
    if (bin_0 >= 0 && bin_0 < p.transient_grad.size(1)) {
      const float transient_grad = p.transient_grad[p.b][bin_0][s][l];
      dloss_dtriangle_part0 +=
        transient_grad * AreaTrapezoid(
          cbin_0, bin_0 + 1,
          cbin_0, y_0,
          cbin_1, y_1
        );
      const float base = bin_0 + 1 - cbin_0;
      dloss_dtriangle_part1 +=
        transient_grad * (
          dy0_dtriangle * base -
          dcbin_0_dtriangle * y_0 -
          dcbin_0_dtriangle * (base * (y_1 - y_0) / (cbin_1 - cbin_0)) +
          (0.5f * base * base) *
            (
              (dy1_dtriangle - dy0_dtriangle) * (cbin_1 - cbin_0) -
              (y_1 - y_0) * (dcbin_1_dtriangle - dcbin_0_dtriangle)
            ) /
            (powf(cbin_1 - cbin_0, 2.f))
        );
    }
    for (int i = bin_0 + 1; i < bin_1; ++i) {
      if (i >= 0 && i < p.transient_grad.size(1)) {
        const float transient_grad = p.transient_grad[p.b][i][s][l];
        dloss_dtriangle_part0 +=
          transient_grad * AreaTrapezoid(
            i, i + 1,
            cbin_0, y_0,
            cbin_1, y_1
          );
        dloss_dtriangle_part1 +=
          transient_grad * (
            dy0_dtriangle -
            dcbin_0_dtriangle * (y_1 - y_0) / (cbin_1 - cbin_0) +
            ((i + 0.5f) - cbin_0) *
              (
                (dy1_dtriangle - dy0_dtriangle) * (cbin_1 - cbin_0) -
                (y_1 - y_0) * (dcbin_1_dtriangle - dcbin_0_dtriangle)
              ) /
              (powf(cbin_1 - cbin_0, 2.f))
          );
      }
    }
    if (bin_1 >= 0 && bin_1 < p.transient_grad.size(1)) {
      const float transient_grad = p.transient_grad[p.b][bin_1][s][l];
      dloss_dtriangle_part0 +=
        transient_grad * AreaTrapezoid(
          bin_1 , cbin_1,
          cbin_0, y_0,
          cbin_1, y_1
        );
      dloss_dtriangle_part1 +=
        transient_grad * (
          dcbin_1_dtriangle * (
            y_0 + (0.5f * (bin_1 + cbin_1) - cbin_0) * (y_1 - y_0) /
            (cbin_1 - cbin_0)) +
          (cbin_1 - bin_1) * (
            dy0_dtriangle +
            (0.5f * dcbin_1_dtriangle - dcbin_0_dtriangle) *
              (y_1 - y_0) / (cbin_1 - cbin_0) +
            (0.5f * (bin_1 + cbin_1) - cbin_0) *
              (
                (dy1_dtriangle - dy0_dtriangle) * (cbin_1 - cbin_0) -
                (y_1 - y_0) * (dcbin_1_dtriangle - dcbin_0_dtriangle)
              ) /
              (powf(cbin_1 - cbin_0, 2.f))
          )
        );
    }
  }
}

/**
 * @brief Evaluate derivative of the normal vector at the given vector
 * 
 * Applies the Jacobian of the normal to the input vector.
 * 
 * @param v1v0 vertex_1 - vertex_0
 * @param v2v0 vertex_2 - vertex_0
 * @param vec  vector to evaluate
 */
__device__ inline vert3 normal_grad_triangle(
    const float3& v1v0,
    const float3& v2v0,
    const float3& vec) {
  return vert3(
      make_float3(
        ( v2v0.z-v1v0.z)*vec.y + (-v2v0.y+v1v0.y)*vec.z,
        (-v2v0.z+v1v0.z)*vec.x + ( v2v0.x-v1v0.x)*vec.z,
        ( v2v0.y-v1v0.y)*vec.x + (-v2v0.x+v1v0.x)*vec.y
      ),
      make_float3(
        (-v2v0.z)*vec.y + ( v2v0.y)*vec.z,
        ( v2v0.z)*vec.x + (-v2v0.x)*vec.z,
        (-v2v0.y)*vec.x + ( v2v0.x)*vec.y
      ),
      make_float3(
        ( v1v0.z)*vec.y + (-v1v0.y)*vec.z,
        (-v1v0.z)*vec.x + ( v1v0.x)*vec.z,
        ( v1v0.y)*vec.x + (-v1v0.x)*vec.y
      )
    );
}

/**
 * @brief Compute light transport between sensor point and triangle
 * 
 * @param vertex_0 First triangle vertex
 * @param vertex_1 Second triangle vertex
 * @param vertex_2 Thrist triangle vertex
 * @param centroid Triangle centroid
 * @param material Triangle material properties
 * @param model Triangle material model
 * @param scan_point Scan and laser point on the wall
 * @param scan_normal Normal at scan_point
 * @param backface_culling Do not render backsides of triangles
 */
__device__ inline float AlphaConfocal(
    const float3& vertex_0, const float3& vertex_1, const float3& vertex_2,
    const float3& centroid, const float4& material, const int model,
    const float3& scan_point, const float3& scan_normal,
    bool backface_culling) {
  const float3 normal = cross(vertex_1 - vertex_0, vertex_2 - vertex_0);
  const float3 scanpoint_centroid_ray = centroid - scan_point;
  const float alpha =
    1.f/length(normal) *
    powf(length(scanpoint_centroid_ray), -8.f) *
    powf(
      max(0.f, dot(scan_normal, scanpoint_centroid_ray)),
      2.f) *
    powf(
      backface_culling ?
        max(0.f, -dot(normal, scanpoint_centroid_ray)) :
        fabsf(dot(normal, scanpoint_centroid_ray)),
      2.f);
  return 1.e2 * alpha * material.x;
}

/**
 * @brief Compute derivative of light transport between sensor point and triangle
 * 
 * @param vertex_0 First triangle vertex
 * @param vertex_1 Second triangle vertex
 * @param vertex_2 Thrist triangle vertex
 * @param centroid Triangle centroid
 * @param material Triangle material properties
 * @param model Triangle material model
 * @param scan_point Scan and laser point on the wall
 * @param scan_normal Normal at scan_point
 * @param backface_culling Do not render backsides of triangles
 * @param alpha Corresponding alpha value
 */
__device__ inline vert3 AlphaConfocalGradVerts(
    const float3& vertex_0, const float3& vertex_1, const float3& vertex_2,
    const float3& centroid, const float4& material, const int model,
    const float3& scan_point, const float3& scan_normal,
    bool backface_culling, const float alpha) {
  vert3 alpha_grad;
  if (alpha == 0.f) {
    return alpha_grad;
  }
  const float3 v1v0 = vertex_1 - vertex_0;
  const float3 v2v0 = vertex_2 - vertex_0;
  const float3 normal = cross(v1v0, v2v0);
  const float3 scanpoint_centroid_ray = centroid - scan_point;
  // 1.f/length(normal)
  alpha_grad += (-1.f / length_sqr(normal)) * normal_grad_triangle(v1v0, v2v0, normal);
  // powf(length(scanpoint_centroid_ray), -8.f)
  alpha_grad += (-8.f /(3.f * length_sqr(scanpoint_centroid_ray))) * vert3(
    scanpoint_centroid_ray,
    scanpoint_centroid_ray,
    scanpoint_centroid_ray
  );
  // powf(dot(scan_normal, scanpoint_centroid_ray) , 2.f)
  alpha_grad += (2.f / (3.f*dot(scan_normal, scanpoint_centroid_ray))) * vert3(
    scan_normal,
    scan_normal,
    scan_normal
  );
  // powf(dot(normal, scanpoint_centroid_ray), 2.f);
  alpha_grad += (2.f / dot(normal, scanpoint_centroid_ray)) * (
    normal_grad_triangle(v1v0, v2v0, scanpoint_centroid_ray) +
    ((1.f/3.f) * vert3(normal, normal, normal))
  );
  // Because of logarithmic derivative
  return alpha * alpha_grad;
}

/**
 * @brief Compute derivative of light transport between sensor point and triangle
 * 
 * @param vertex_0 First triangle vertex
 * @param vertex_1 Second triangle vertex
 * @param vertex_2 Thrist triangle vertex
 * @param centroid Triangle centroid
 * @param material Triangle material properties
 * @param model Triangle material model
 * @param scan_point Scan and laser point on the wall
 * @param scan_normal Normal at scan_point
 * @param backface_culling Do not render backsides of triangles
 * @param alpha Corresponding alpha value
 */
__device__ inline float4 AlphaConfocalGradMaterial(
    const float3& vertex_0, const float3& vertex_1, const float3& vertex_2,
    const float3& centroid, const float4& material, const int model,
    const float3& scan_point, const float3& scan_normal,
    bool backface_culling, const float alpha) {
  float4 material_grad = make_float4(0.f, 0.f, 0.f, 0.f);
  if (alpha <= 0.f) {
    return material_grad;
  }
  material_grad.x = alpha / material.x;
  return material_grad;
}

/**
 * @brief Compute light transport between sensor point and triangle
 * 
 * @param vertex_0 First triangle vertex
 * @param vertex_1 Second triangle vertex
 * @param vertex_2 Thrist triangle vertex
 * @param centroid Triangle centroid
 * @param scan_point Scan point on the wall
 * @param scan_normal Normal at scan_point
 * @param laser_point Laser point on the wall
 * @param laser_normal Normal at laser_point
 * @param backface_culling Do not render backsides of triangles
 */
__device__ inline float AlphaExhaustive(
    const float3& vertex_0, const float3& vertex_1, const float3& vertex_2,
    const float3& centroid, const float4& material, const int model,
    const float3& scan_point, const float3& scan_normal,
    const float3& laser_point, const float3& laser_normal,
    bool backface_culling) {
  const float3 normal = cross(vertex_1 - vertex_0, vertex_2 - vertex_0);
  const float3 scanpoint_centroid_ray = centroid - scan_point;
  const float3 laserpoint_centroid_ray = centroid - laser_point;
  const float alpha =
    1.f/length(normal) *
    powf(length(scanpoint_centroid_ray), -4.f) *
    powf(length(laserpoint_centroid_ray), -4.f) *
    max(0.f, dot(scan_normal, scanpoint_centroid_ray)) *
    max(0.f, dot(laser_normal, laserpoint_centroid_ray)) *
    (backface_culling ? max(0.f, -dot(normal, scanpoint_centroid_ray))
                      : fabsf(dot(normal, scanpoint_centroid_ray))) *
    (backface_culling ? max(0.f, -dot(normal, laserpoint_centroid_ray))
                      : fabsf(dot(normal, laserpoint_centroid_ray)));
  // const float alpha =
  //   length(normal) *
  //   powf(length(scanpoint_centroid_ray), -2.f) *
  //   powf(length(laserpoint_centroid_ray), -2.f) *
  //   max(0.f,  dot(scan_normal, scanpoint_centroid_ray)) *
  //   max(0.f,  dot(laser_normal, laserpoint_centroid_ray));
  return 1.e2 * alpha * material.x;
}

/**
 * @brief Compute derivative of light transport between sensor point and triangle
 * 
 * @param vertex_0 First triangle vertex
 * @param vertex_1 Second triangle vertex
 * @param vertex_2 Thrist triangle vertex
 * @param centroid Triangle centroid
 * @param material Triangle material properties
 * @param model Triangle material model
 * @param scan_point Scan point on the wall
 * @param scan_normal Normal at scan_point
 * @param laser_point Laser point on the wall
 * @param laser_normal Normal at laser_point
 * @param backface_culling Do not render backsides of triangles
 * @param alpha Corresponding alpha value
 */
__device__ inline vert3 AlphaExhaustiveGradVerts(
    const float3& vertex_0, const float3& vertex_1, const float3& vertex_2,
    const float3& centroid, const float4& material, const int model,
    const float3& scan_point, const float3& scan_normal,
    const float3& laser_point, const float3& laser_normal,
    bool backface_culling, const float alpha) {
  vert3 alpha_grad;
  if (alpha == 0.f) {
    return alpha_grad;
  }
  const float3 v1v0 = vertex_1 - vertex_0;
  const float3 v2v0 = vertex_2 - vertex_0;
  const float3 normal = cross(v1v0, v2v0);
  const float3 scanpoint_centroid_ray = centroid - scan_point;
  const float3 laserpoint_centroid_ray = centroid - laser_point;
  // 1.f/length(normal)
  alpha_grad += (-1.f / length_sqr(normal)) * normal_grad_triangle(v1v0, v2v0, normal);
  // powf(length(scanpoint_centroid_ray), -4.f)
  alpha_grad += (-4.f /(3.f * length_sqr(scanpoint_centroid_ray))) * vert3(
    scanpoint_centroid_ray,
    scanpoint_centroid_ray,
    scanpoint_centroid_ray
  );
  // powf(length(laserpoint_centroid_ray), -4.f)
  alpha_grad += (-4.f /(3.f * length_sqr(laserpoint_centroid_ray))) * vert3(
    laserpoint_centroid_ray,
    laserpoint_centroid_ray,
    laserpoint_centroid_ray
  );
  // dot(scan_normal, scanpoint_centroid_ray)
  alpha_grad += (1.f / (3.f*dot(laser_normal, scanpoint_centroid_ray))) * vert3(
    laser_normal,
    laser_normal,
    laser_normal
  );
  // dot(laser_normal, laserpoint_centroid_ray)
  alpha_grad += (1.f / (3.f*dot(scan_normal, laserpoint_centroid_ray))) * vert3(
    scan_normal,
    scan_normal,
    scan_normal
  );
  // fabsf(dot(normal, scanpoint_centroid_ray))
  float dot_normal_scanpoint_centroid_ray = dot(normal, scanpoint_centroid_ray);
  alpha_grad += (signt(dot_normal_scanpoint_centroid_ray) / fabsf(dot_normal_scanpoint_centroid_ray)) * (
    normal_grad_triangle(v1v0, v2v0, scanpoint_centroid_ray) +
    ((1.f/3.f) * vert3(normal, normal, normal))
  );
  // dot(normal, laserpoint_centroid_ray)
  float dot_normal_laserpoint_centroid_ray = dot(normal, laserpoint_centroid_ray);
  alpha_grad += (signt(dot_normal_laserpoint_centroid_ray) / fabsf(dot_normal_laserpoint_centroid_ray)) * (
    normal_grad_triangle(v1v0, v2v0, laserpoint_centroid_ray) +
    ((1.f/3.f) * vert3(normal, normal, normal))
  );
  // // length(normal)
  // alpha_grad += (1.f / length(normal)) * normal_grad_triangle(v1v0, v2v0, normal);
  // // powf(length(scanpoint_centroid_ray), -2.f)
  // alpha_grad += (-2.f /(3.f * length_sqr(scanpoint_centroid_ray))) * vert3(
  //   scanpoint_centroid_ray,
  //   scanpoint_centroid_ray,
  //   scanpoint_centroid_ray
  // );
  // // powf(length(laserpoint_centroid_ray), -2.f)
  // alpha_grad += (-2.f /(3.f * length_sqr(laserpoint_centroid_ray))) * vert3(
  //   laserpoint_centroid_ray,
  //   laserpoint_centroid_ray,
  //   laserpoint_centroid_ray
  // );
  // // dot(scan_normal, scanpoint_centroid_ray)
  // alpha_grad += (1.f / (3.f*dot(laser_normal, scanpoint_centroid_ray))) * vert3(
  //   laser_normal,
  //   laser_normal,
  //   laser_normal
  // );
  // // dot(laser_normal, laserpoint_centroid_ray)
  // alpha_grad += (1.f / (3.f*dot(scan_normal, laserpoint_centroid_ray))) * vert3(
  //   scan_normal,
  //   scan_normal,
  //   scan_normal
  // );
  // Because of logarithmic derivative
  return alpha * alpha_grad;
}

/**
 * @brief Compute derivative of light transport between sensor point and triangle
 * 
 * @param vertex_0 First triangle vertex
 * @param vertex_1 Second triangle vertex
 * @param vertex_2 Thrist triangle vertex
 * @param centroid Triangle centroid
 * @param material Triangle material properties
 * @param model Triangle material model
 * @param scan_point Scan point on the wall
 * @param scan_normal Normal at scan_point
 * @param laser_point Laser point on the wall
 * @param laser_normal Normal at laser_point
 * @param backface_culling Do not render backsides of triangles
 * @param alpha Corresponding alpha value
 */
__device__ inline float4 AlphaExhaustiveGradMaterial(
    const float3& vertex_0, const float3& vertex_1, const float3& vertex_2,
    const float3& centroid, const float4& material, const int model,
    const float3& scan_point, const float3& scan_normal,
    const float3& laser_point, const float3& laser_normal,
    bool backface_culling, const float alpha) {
  float4 material_grad = make_float4(0.f, 0.f, 0.f, 0.f);
  if (alpha <= 0.f) {
    return material_grad;
  }
  material_grad.x = alpha / material.x;
  return material_grad;
}

} // namespace render
} // namespace totri

#endif // TOTRI_RENDER_PROGRAM_MESH_COMMON_H
