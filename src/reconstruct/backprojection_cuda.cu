#include "reconstruct/backprojection_cuda.h"
#include <cuda_runtime.h>

#include "common/const.h"
#include "common/math.h"

namespace totri {
namespace reconstruct {

template <typename scalar_t>
__global__ void backprojection_exhaustive_arellano_forward_kernel(
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> transient, // [B, N, S, L]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> scan_points, // [B, 3, S]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> laser_points, // [B, 3, L]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> scan_origin, // [B, 3, S]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> laser_origin, // [B, 3, L]
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> volume,  // [B, D, H, W]
    const float3 volume_start, const float3 volume_end,
    const float bin_width, const float bin_offset,
    const int b) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x; // Voxel
  const int s = blockIdx.y * blockDim.y + threadIdx.y; // Scan point
  const int l = blockIdx.z * blockDim.z + threadIdx.z; // Laser pos
  if (s < scan_points.size(2) && l < laser_points.size(2) && v < volume.size(2)*volume.size(3)) {
    const bool use_scan_dist = scan_origin.size(2) > 0;
    const bool use_laser_dist = laser_origin.size(2) > 0;
    const int xi = v % volume.size(3);
    const int yi = v / volume.size(3);
    const float xf = volume_start.x +
      (float(xi) / float(volume.size(3) - 1)) * (volume_end.x - volume_start.x);
    const float yf = volume_start.y +
      (float(yi) / float(volume.size(2) - 1)) * (volume_end.y - volume_start.y);
    // Scan position on the wall
    const float3 scan_point = make_float3(
      scan_points[b][0][s],
      scan_points[b][1][s],
      scan_points[b][2][s]);
    const scalar_t scan_point_dist = (scan_origin.size(2) > 0) ?
      distance(scan_point, make_float3(
        scan_origin[b][0][s],
        scan_origin[b][1][s],
        scan_origin[b][2][s])) :
      scalar_t(0.0);
    // Laser position on the wall
    const float3 laser_point = make_float3(
      laser_points[b][0][l],
      laser_points[b][1][l],
      laser_points[b][2][l]);
    const scalar_t laser_point_dist = (laser_origin.size(2) > 0) ?
      distance(laser_point, make_float3(
        laser_origin[b][0][l],
        laser_origin[b][1][l],
        laser_origin[b][2][l])) :
      scalar_t(0.0);
    // Center of the ellipsoid
    const float3 center = make_float3(
      0.5f*(scan_point.x + laser_point.x),
      0.5f*(scan_point.y + laser_point.y),
      0.5f*(scan_point.z + laser_point.z)
    );
    // Rotation axis of ellipsoid
    float3 normal = make_float3(
      scan_point.x - laser_point.x,
      scan_point.y - laser_point.y,
      scan_point.z - laser_point.z
    );
    const float length = norm3df(normal.x, normal.y, normal.z);
    if (length > 1.e-6) {
      normal = (1.f/length) * normal;
    }
    // Start of the ray through the volume at xy
    const float3 ray_start = make_float3(xf, yf, volume_start.z) - center;
    // End of the ray through the volume at xy
    const float3 ray_end = make_float3(xf, yf, volume_end.z) - center;
    for (int ti=0; ti < transient.size(1); ++ti) {
      const float value = transient[b][ti][s][l];
      if (value > 0) {
        // Distance corresponding to bin ti
        const scalar_t tc = (scalar_t(ti) * bin_width + bin_offset)
          - scan_point_dist - laser_point_dist;
        if (tc > length) {
          // Radii
          const float r0 = tc/scalar_t(2);
          const float r1 = sqrtt(powt(tc/scalar_t(2),scalar_t(2)) - length*length/4.f);
          const float scale = r1 / r0;
          // Transform points
          const float3 ray_start_t = ray_start + (scale - 1.f) * dot(ray_start, normal) * normal;
          const float3 ray_end_t = ray_end + (scale - 1.f) * dot(ray_end, normal) * normal;
          float3 ray_direction = ray_end_t - ray_start_t;
          const float ray_length = norm3df(ray_direction.x, ray_direction.y, ray_direction.z);
          ray_direction = (1.f/ray_length)*ray_direction;
          // Intersect with sphere around center with radius r1
          const float v0 = dot(ray_direction, ray_start_t);
          float v1 = v0*v0 - (dot(ray_start_t, ray_start_t) - r1*r1);
          if (v1 > 0) {
            v1 = sqrtf(v1);
            float d = -v0 - v1;
            if (d < 0) {
              d = -v0 + v1;
            }
            int binf = d / ray_length * float(volume.size(1)) + 0.5f;
            int bini = floorf(binf);
            float alpha = binf - float(bini);
            if (bini >= 0 && bini < volume.size(1)) {
              atomicAdd(&volume[b][bini  ][yi][xi], (1.f-alpha) * value);
            }
            if (bini+1 >= 0 && bini+1 < volume.size(1)) {
              atomicAdd(&volume[b][bini+1][yi][xi],       alpha * value);
            }
          }
        }
      }
    }
  }
}

void backprojection_exhaustive_arellano_forward_cuda(
    const at::Tensor transient, // [B, N, S, L]
    const at::Tensor scan_points, // [B, 3, S]
    const at::Tensor laser_points, // [B, 3, L]
    const at::Tensor scan_origin, // [B, 3, S]
    const at::Tensor laser_origin, // [B, 3, L]
    at::Tensor volume, // [B, D, H, W]
    const std::array<float, 3>& volume_start, const std::array<float, 3>& volume_end,
    const float bin_width, const float bin_offset) {
  // Conversions
  const float3 volume_start_f = make_float3(volume_start[0], volume_start[1], volume_start[2]);
  const float3 volume_end_f = make_float3(volume_end[0], volume_end[1], volume_end[2]);
  // Dims
  const dim3 threads(64, 1, 1);
  const dim3 blocks(
    (volume.size(3)*volume.size(2) + threads.x - 1) / threads.x,
    (scan_points.size(2)           + threads.y - 1) / threads.y,
    (laser_points.size(2)          + threads.z - 1) / threads.z);
  for (int b=0; b < transient.size(0); ++b) {
    AT_DISPATCH_FLOATING_TYPES(
      transient.type(),
      "backprojection_exhaustive_arellano_forward_kernel",
      ([&] {backprojection_exhaustive_arellano_forward_kernel<scalar_t><<<blocks, threads>>>(
        transient.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        scan_points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        laser_points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        scan_origin.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        laser_origin.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        volume.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        volume_start_f, volume_end_f, bin_width, bin_offset, b
      );
    }));
  }
}

template <typename scalar_t>
__global__ void backprojection_confocal_velten_forward_kernel(
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> transient, // [B, T, S]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> scan_points, // [B, S, 3]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> scan_origin, // [B, S, 3]
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> volume,  // [B, D, H, W]
    const float3 volume_start, const float3 volume_end,
    const float bin_width, const float bin_offset,
    const int b) {
  // Voxel index
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < volume.size(3) && y < volume.size(2) && z < volume.size(1)) {
    scalar_t value = 0.f;
    const float3 volume_point = make_float3(
      volume_start.x + (float(x) / float(volume.size(3) - 1)) * (volume_end.x - volume_start.x),
      volume_start.y + (float(y) / float(volume.size(2) - 1)) * (volume_end.y - volume_start.y),
      volume_start.z + (float(z) / float(volume.size(1) - 1)) * (volume_end.z - volume_start.z));
    for (int s=0; s<scan_points.size(1); ++s) {
      // Scan position on the wall
      const float3 scan_point = make_float3(
        scan_points[b][s][0],
        scan_points[b][s][1],
        scan_points[b][s][2]);
      const scalar_t scan_volume_dist = distance(scan_point, volume_point);
      const scalar_t scan_origin_dist = (scan_origin.size(1) > 0) ?
        distance(scan_point, make_float3(
          scan_origin[b][s][0],
          scan_origin[b][s][1],
          scan_origin[b][s][2])) :
        scalar_t(0.0);
      // Full distance
      const scalar_t dist = 2 * scan_origin_dist + 2 * scan_volume_dist;
      // Corresponding bin
      const int ti = floorf((dist - bin_offset) / bin_width + 0.5f);
      if (ti >= 0 && ti < transient.size(1)) {
        value += transient[b][ti][s];
      }
    }
    volume[b][z][y][x] = value;
  }
}

void backprojection_confocal_velten_forward_cuda(
    const at::Tensor transient, // [B, T, S]
    const at::Tensor scan_points, // [B, S, 3]
    const at::Tensor scan_origin, // [B, S, 3]
    at::Tensor volume, // [B, D, H, W]
    const std::array<float, 3>& volume_start,
    const std::array<float, 3>& volume_end,
    const float bin_width, const float bin_offset) {
  // Conversions
  const float3 volume_start_f = make_float3(volume_start[0], volume_start[1], volume_start[2]);
  const float3 volume_end_f = make_float3(volume_end[0], volume_end[1], volume_end[2]);
  // Dims
  const dim3 threads(8, 8, 8);
  const dim3 blocks(
    (volume.size(3) + threads.x - 1) / threads.x,
    (volume.size(2) + threads.y - 1) / threads.y,
    (volume.size(1) + threads.z - 1) / threads.z);
  for (int b=0; b < transient.size(0); ++b) {
    AT_DISPATCH_FLOATING_TYPES(
      transient.type(),
      "backprojection_confocal_velten_forward_kernel",
      ([&] {backprojection_confocal_velten_forward_kernel<scalar_t><<<blocks, threads>>>(
        transient.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        scan_points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        scan_origin.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        volume.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        volume_start_f, volume_end_f, bin_width, bin_offset, b
      );
    }));
  }
}

template <typename scalar_t>
__global__ void backprojection_exhaustive_velten_forward_kernel(
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> transient, // [B, N, S, L]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> scan_points,  // [B, S, 3]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> laser_points, // [B, L, 3]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> scan_origin,  // [B, S, 3]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> laser_origin, // [B, L, 3]
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> volume,  // [B, D, H, W]
    const float3 volume_start, const float3 volume_end,
    const float bin_width, const float bin_offset,
    const int b) {
  // Voxel index
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < volume.size(3) && y < volume.size(2) && z < volume.size(1)) {
    scalar_t value = 0.f;
    const float3 volume_point = make_float3(
      volume_start.x + (float(x) / float(volume.size(3) - 1)) * (volume_end.x - volume_start.x),
      volume_start.y + (float(y) / float(volume.size(2) - 1)) * (volume_end.y - volume_start.y),
      volume_start.z + (float(z) / float(volume.size(1) - 1)) * (volume_end.z - volume_start.z));
    for (int s=0; s<scan_points.size(1); ++s) {
      // Scan position on the wall
      const float3 scan_point = make_float3(
        scan_points[b][s][0],
        scan_points[b][s][1],
        scan_points[b][s][2]);
      const scalar_t scan_volume_dist = distance(scan_point, volume_point);
      const scalar_t scan_origin_dist = (scan_origin.size(1) > 0) ?
        distance(scan_point, make_float3(
          scan_origin[b][s][0],
          scan_origin[b][s][1],
          scan_origin[b][s][2])) :
        scalar_t(0.0);
      for (int l=0; l<laser_points.size(1); ++l) {
        // Laser position on the wall
        const float3 laser_point = make_float3(
          laser_points[b][l][0],
          laser_points[b][l][1],
          laser_points[b][l][2]);
        const scalar_t laser_volume_dist = distance(laser_point, volume_point);
        const scalar_t laser_origin_dist = (laser_origin.size(1) > 0) ?
          distance(laser_point, make_float3(
            laser_origin[b][l][0],
            laser_origin[b][l][1],
            laser_origin[b][l][2])) :
          scalar_t(0.0);
        // Full distance
        const scalar_t dist = scan_origin_dist + scan_volume_dist +
          laser_volume_dist + laser_origin_dist;
        // Corresponding bin
        const int ti = floorf((dist - bin_offset) / bin_width + 0.5f);
        if (ti >= 0 && ti < transient.size(1)) {
          value += transient[b][ti][s][l];
        }
      }
    }
    volume[b][z][y][x] = value;
  }
}

void backprojection_exhaustive_velten_forward_cuda(
    const at::Tensor transient, // [B, N, S, L]
    const at::Tensor scan_points,  // [B, S, 3]
    const at::Tensor laser_points, // [B, L, 3]
    const at::Tensor scan_origin,  // [B, S, 3]
    const at::Tensor laser_origin, // [B, L, 3]
    at::Tensor volume, // [B, D, H, W]
    const std::array<float, 3>& volume_start, const std::array<float, 3>& volume_end,
    const float bin_width, const float bin_offset) {
  // Conversions
  const float3 volume_start_f = make_float3(volume_start[0], volume_start[1], volume_start[2]);
  const float3 volume_end_f = make_float3(volume_end[0], volume_end[1], volume_end[2]);
  // Dims
  const dim3 threads(8, 8, 8);
  const dim3 blocks(
    (volume.size(3) + threads.x - 1) / threads.x,
    (volume.size(2) + threads.y - 1) / threads.y,
    (volume.size(1) + threads.z - 1) / threads.z);
  for (int b=0; b < transient.size(0); ++b) {
    AT_DISPATCH_FLOATING_TYPES(
      transient.type(),
      "backprojection_exhaustive_velten_forward_kernel",
      ([&] {backprojection_exhaustive_velten_forward_kernel<scalar_t><<<blocks, threads>>>(
        transient.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        scan_points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        laser_points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        scan_origin.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        laser_origin.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        volume.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        volume_start_f, volume_end_f, bin_width, bin_offset, b
      );
    }));
  }
}

} // namespace reconstruct
} // namespace totri
