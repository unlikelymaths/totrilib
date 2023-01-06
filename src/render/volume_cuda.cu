#include "render/volume_cuda.h"

#include <cuda_runtime.h>

#include "common/const.h"
#include "common/math.h"

namespace totri {
namespace render {

template <typename scalar_t>
__global__ void volume_confocal_forward_kernel(
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> volume,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> measurement_points,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> transient,
    const float3 volume_origin, const float3 volume_size,
    const float bin_width, const float bin_offset) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x; // Measurement Point
  const int ti = blockIdx.y * blockDim.y + threadIdx.y; // Time Bin
  // Find all times for this point
  if (i < transient.size(0) && ti < transient.size(1)){
    // Distance corresponding to bin t
    const scalar_t tc = ti * bin_width + bin_offset;
    // Coordinates of the measurement point on the wall
    const scalar_t wx = measurement_points[0][i];
    const scalar_t wy = measurement_points[1][i];
    const scalar_t wz = measurement_points[2][i];
    // z offset and scaling
    const scalar_t o = volume_origin.z;
    const scalar_t s = scalar_t(volume.size(2)) / scalar_t(volume_size.z);
    // Resulting Value
    scalar_t value = 0;
    for (int xi=0; xi<volume.size(0); ++xi) {
      // Corresponding x position in the volume
      const scalar_t x = scalar_t(volume_origin.x) + (xi + scalar_t(0.5)) * scalar_t(volume_size.x) / scalar_t(volume.size(0));
      for (int yi=0; yi<volume.size(1); ++yi) {
        // Corresponding y position in the volume
        const scalar_t y = scalar_t(volume_origin.y) + (yi + scalar_t(0.5)) * scalar_t(volume_size.y) / scalar_t(volume.size(1));
        // Distance of the intersection point from wz squared
        const scalar_t depth_squared = powt(tc, scalar_t(2)) - powt(wx - x, scalar_t(2)) - powt(wy - y, scalar_t(2));
        if (depth_squared > 0) {
          const scalar_t depth = sqrtt(depth_squared);
          // z coordinate of the intersection
          const scalar_t z = wz + depth;
          // Corresponding bin
          const scalar_t z_hat = (z - o) * s - scalar_t(0.5);
          const int zi = floorf(z_hat);
          // Only continue if we are inside the volume and not on the boundary
          if (zi >= 0 && zi + 1 < volume.size(2)) {
            // Weight to interpolate between both bins
            const scalar_t alpha = z_hat - scalar_t(zi);
            // f1 (interpolation of the volume)
            const scalar_t volume_xi_yi_zi0 = volume[xi][yi][zi];
            const scalar_t volume_xi_yi_zi1 = volume[xi][yi][zi+1];
            const scalar_t f1 = (1 - alpha) * volume_xi_yi_zi0 + alpha * volume_xi_yi_zi1;
            // f2 (lighting)
            const scalar_t f2 = powt(-depth / powt(tc, scalar_t(2)), scalar_t(4));
            // Final result
            value += f1 * f2;
          }
        }
      }
    }
    transient[i][ti] = value;
  }
}

void volume_confocal_forward_cuda(
    const at::Tensor volume, // X x Y x Z
    const at::Tensor measurement_points, // 3 x N
    at::Tensor transient, // N x T
    const std::array<float, 3>& volume_origin, const std::array<float, 3>& volume_size,
    const float bin_width, const float bin_offset) {
  // Conversions
  const float3 volume_origin_f = make_float3(volume_origin[0], volume_origin[1], volume_origin[2]);
  const float3 volume_size_f = make_float3(volume_size[0], volume_size[1], volume_size[2]);
  // Dims
  auto shape = transient.sizes();
  const dim3 threads(8, 8, 1);
  const dim3 blocks(
    (shape[0] + threads.x - 1) / threads.x,
    (shape[1] + threads.y - 1) / threads.y,
    1
  );
  AT_DISPATCH_FLOATING_TYPES(volume.type(), "volume_confocal_forward_kernel", ([&] {
    volume_confocal_forward_kernel<scalar_t><<<blocks, threads>>>(
      volume.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      measurement_points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
      transient.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
      volume_origin_f, volume_size_f, bin_width, bin_offset
    );
  }));
}

template <typename scalar_t>
__global__ void volume_confocal_grad_volume_kernel(
    at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> volume,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> measurement_points,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> transient,
    float3 volume_origin, float3 volume_size,
    float bin_width, float bin_offset) {
  const int xi = blockIdx.x * blockDim.x + threadIdx.x;
  const int yi = blockIdx.y * blockDim.y + threadIdx.y;
  const int zi = blockIdx.z * blockDim.z + threadIdx.z;
  if (xi < volume.size(0) && yi < volume.size(1) && zi < volume.size(2)){
    // volume Position
    const scalar_t x = scalar_t(volume_origin.x) + (scalar_t(xi) + scalar_t(0.5)) * scalar_t(volume_size.x) / scalar_t(volume.size(0));
    const scalar_t y = scalar_t(volume_origin.y) + (scalar_t(yi) + scalar_t(0.5)) * scalar_t(volume_size.y) / scalar_t(volume.size(1));
    const scalar_t z = scalar_t(volume_origin.z) + (scalar_t(zi) + scalar_t(0.5)) * scalar_t(volume_size.z) / scalar_t(volume.size(2));
    // z offset and scaling
    const scalar_t o = volume_origin.z;
    const scalar_t s = scalar_t(volume.size(2)) / scalar_t(volume_size.z);
    // Z scaling
    const scalar_t dz = scalar_t(volume_size.z) / scalar_t(volume.size(2));
    // Resulting volume intensity
    scalar_t value = 0;
    for (int i=0; i<measurement_points.size(1); ++i) {
      // Coordinates of the measurement point on the wall
      const scalar_t wx = measurement_points[0][i];
      const scalar_t wy = measurement_points[1][i];
      const scalar_t wz = measurement_points[2][i];
      // Distance from the previous and next volume in the volume to measurement location
      const scalar_t zz = zi > 0 ? dz : 0;
      const scalar_t distance_plane_squared = powt(x - wx, scalar_t(2)) + powt(y - wy, scalar_t(2));
      const scalar_t distance_start = sqrtt(distance_plane_squared + powt(z - zz - wz, scalar_t(2)));
      const scalar_t distance_mid =   sqrtt(distance_plane_squared + powt(z - wz     , scalar_t(2)));
      const scalar_t distance_end =   sqrtt(distance_plane_squared + powt(z + dz - wz, scalar_t(2)));
      // Bin in the histogram corresponding to time
      const scalar_t tf_start = (distance_start - scalar_t(bin_offset)) / scalar_t(bin_width);
      const scalar_t tf_mid   = (distance_mid   - scalar_t(bin_offset)) / scalar_t(bin_width);
      const scalar_t tf_end   = (distance_end   - scalar_t(bin_offset)) / scalar_t(bin_width);
      const int t_start = min(max(int(ceilf(tf_start)), 0), transient.size(1));
      const int t_mid   = min(max(int(ceilf(tf_mid  )), 0), transient.size(1));
      const int t_end   = min(max(int(ceilf(tf_end  )), 0), transient.size(1));
      for (int ti=t_start; ti<t_end; ++ti) {
        // Distance corresponding to bin t
        const scalar_t tc = ti * bin_width + bin_offset;
        // Distance of the intersection point from wz squared
        const scalar_t depth_squared = powt(tc, scalar_t(2)) - distance_plane_squared;
        if (depth_squared > 0) {
          const scalar_t depth = sqrtt(depth_squared);
          // z coordinate of the intersection
          const scalar_t z = wz + depth;
          // Corresponding bin
          const scalar_t z_hat = (z - o) * s - scalar_t(0.5);
          const int zi = floorf(z_hat);
          // Weight to interpolate between both bins
          const scalar_t alpha = z_hat - scalar_t(zi);
          // f1 (interpolation of the volume)
          const scalar_t df1dt = (ti >= t_mid) ? (1 - alpha) : (alpha);
          // f2 (lighting)
          const scalar_t f2 = powt(-depth / powt(tc, scalar_t(2)), scalar_t(4));
          // Final result
          value += df1dt * f2 * transient[i][ti];
        }
      }
    }
    volume[xi][yi][zi] = value;
  }
}
void volume_confocal_grad_volume_cuda(
    at::Tensor volume, // X x Y x Z
    at::Tensor measurement_points, // 3 x N
    at::Tensor transient, // N x T
    const std::array<float, 3>& volume_origin, const std::array<float, 3>& volume_size,
    float bin_width, float bin_offset) {
  // Conversions
  float3 volume_origin_f = make_float3(volume_origin[0], volume_origin[1], volume_origin[2]);
  float3 volume_size_f = make_float3(volume_size[0], volume_size[1], volume_size[2]);
  // Dims
  auto shape = volume.sizes();
  const dim3 threads(8, 8, 8);
  const dim3 blocks(
    (shape[0] + threads.x - 1) / threads.x,
    (shape[1] + threads.y - 1) / threads.y,
    (shape[2] + threads.z - 1) / threads.z
  );
  AT_DISPATCH_FLOATING_TYPES(volume.type(), "volume_confocal_grad_volume_kernel", ([&] {
    volume_confocal_grad_volume_kernel<scalar_t><<<blocks, threads>>>(
      volume.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      measurement_points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
      transient.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
      volume_origin_f, volume_size_f, bin_width, bin_offset
    );
  }));
}

template <typename scalar_t>
__global__ void volume_confocal_grad_measurement_points_kernel(
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> measurement_points_grad,
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> volume,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> measurement_points,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> transient_grad,
    float3 volume_origin, float3 volume_size,
    float bin_width, float bin_offset) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x; // Measurement Point
  const int ti = blockIdx.y * blockDim.y + threadIdx.y; // Time Bin
  // Find all times for this point
  if (i < transient_grad.size(0) && ti < transient_grad.size(1)){
    // Incoming gradient
    const scalar_t dl_di = transient_grad[i][ti];
    // Distance corresponding to bin t
    const scalar_t tc = ti * bin_width + bin_offset;
    // Coordinates of the measurement point on the wall
    const scalar_t wx = measurement_points[0][i];
    const scalar_t wy = measurement_points[1][i];
    const scalar_t wz = measurement_points[2][i];
    // z offset and scaling
    const scalar_t o = volume_origin.z;
    const scalar_t s = scalar_t(volume.size(2)) / scalar_t(volume_size.z);
    // Resulting Gradient w.r.t. the measurement point
    scalar_t di_dwx = 0;
    scalar_t di_dwy = 0;
    scalar_t di_dwz = 0;
    for (int xi=0; xi<volume.size(0); ++xi) {
      // Corresponding x position in the volume
      const scalar_t x = scalar_t(volume_origin.x) + (xi + scalar_t(0.5)) * scalar_t(volume_size.x) / scalar_t(volume.size(0));
      for (int yi=0; yi<volume.size(1); ++yi) {
        // Corresponding y position in the volume
        const scalar_t y = scalar_t(volume_origin.y) + (yi + scalar_t(0.5)) * scalar_t(volume_size.y) / scalar_t(volume.size(1));
        // Distance of the intersection point from wz squared
        const scalar_t depth_squared = powt(tc, scalar_t(2)) - powt(wx - x, scalar_t(2)) - powt(wy - y, scalar_t(2));
        if (depth_squared > 0) {
          const scalar_t depth = sqrtt(depth_squared);
          // z coordinate of the intersection
          const scalar_t z = wz + depth;
          // Derivatives
          const scalar_t dz_dwx = - (wx - x) / depth;
          const scalar_t dz_dwy = - (wy - y) / depth;
          const scalar_t dz_dwz = 1;
          // Corresponding bin
          scalar_t z_hat = (z - o) * s - scalar_t(0.5);
          int zi = floorf(z_hat);
          // Only continue if we are inside the volume and not on the boundary
          if (zi >= 0 && zi + 1 < volume.size(2)) {
            // Weight to interpolate between both bins
            const scalar_t alpha = z_hat - scalar_t(zi);
            // f1 (interpolation of the volume)
            const scalar_t volume_xi_yi_zi0 = volume[xi][yi][zi];
            const scalar_t volume_xi_yi_zi1 = volume[xi][yi][zi+1];
            const scalar_t f1 = (1 - alpha) * volume_xi_yi_zi0 + alpha * volume_xi_yi_zi1;
            // Derivatives
            const scalar_t df1_dwx = s * dz_dwx * (volume_xi_yi_zi1 - volume_xi_yi_zi0);
            const scalar_t df1_dwy = s * dz_dwy * (volume_xi_yi_zi1 - volume_xi_yi_zi0);
            const scalar_t df1_dwz = s * dz_dwz * (volume_xi_yi_zi1 - volume_xi_yi_zi0);
            // f2 (lighting)
            const scalar_t f2 = powt(-depth / powt(tc, scalar_t(2)), scalar_t(4));
            // Derivatives
            const scalar_t tmp = scalar_t(4) * powt(-depth / powt(tc, scalar_t(8.0/3.0)), scalar_t(3));
            const scalar_t df2_dwx = -dz_dwx * tmp;
            const scalar_t df2_dwy = -dz_dwy * tmp;
            const scalar_t df2_dwz = (1 - dz_dwz) * tmp;
            // Final result
            di_dwx += f1 * df2_dwx + df1_dwx * f2;
            di_dwy += f1 * df2_dwy + df1_dwy * f2;
            di_dwz += f1 * df2_dwz + df1_dwz * f2;
          }
        }
      }
    }
    atomicAdd(&measurement_points_grad[0][i], dl_di * di_dwx);
    atomicAdd(&measurement_points_grad[1][i], dl_di * di_dwy);
    atomicAdd(&measurement_points_grad[2][i], dl_di * di_dwz);
  }
}

void volume_confocal_grad_measurement_points_cuda(
    at::Tensor measurement_points_grad, // 3 x N
    const at::Tensor volume, // X x Y x Z
    const at::Tensor measurement_points, // 3 x N
    const at::Tensor transient_grad, // N x T
    const std::array<float, 3>& volume_origin, const std::array<float, 3>& volume_size,
    float bin_width, float bin_offset) {
  // Conversions
  float3 volume_origin_f = make_float3(volume_origin[0], volume_origin[1], volume_origin[2]);
  float3 volume_size_f = make_float3(volume_size[0], volume_size[1], volume_size[2]);
  // Dims
  auto shape = transient_grad.sizes();
  const dim3 threads(8, 8, 1);
  const dim3 blocks(
    (shape[0] + threads.x - 1) / threads.x,
    (shape[1] + threads.y - 1) / threads.y,
    1
  );
  measurement_points_grad.fill_(0);
  AT_DISPATCH_FLOATING_TYPES(volume.type(), "volume_confocal_grad_measurement_points_kernel", ([&] {
    volume_confocal_grad_measurement_points_kernel<scalar_t><<<blocks, threads>>>(
      measurement_points_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
      volume.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
      measurement_points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
      transient_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
      volume_origin_f, volume_size_f, bin_width, bin_offset
    );
  }));
}

template <typename scalar_t>
__global__ void volume_render_exhaustive_forward_kernel(
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> volume, // [B, D, H, W]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> scan_points, // [B, 3, S]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> laser_points, // [B, 3, L]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> scan_origin, // [B, 3, S]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> laser_origin, // [B, 3, L]
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> transient,  // [B, N, S, L]
    const float3 volume_start, const float3 volume_end,
    const float bin_width, const float bin_offset,
    const int b) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x; // Voxel
  const int s = blockIdx.y * blockDim.y + threadIdx.y; // Scan point
  const int l = blockIdx.z * blockDim.z + threadIdx.z; // Laser pos
  if (s < scan_points.size(2) && l < laser_points.size(2) && v < volume.size(2)*volume.size(3)) {
    // const bool use_scan_dist = scan_origin.size(2) > 0;
    // const bool use_laser_dist = laser_origin.size(2) > 0;
    const int xi = v % volume.size(3);
    const int yi = v / volume.size(3);
    const float xf = volume_start.x +
      (float(xi) + 0.5f) / float(volume.size(3)) * (volume_end.x - volume_start.x);
    const float yf = volume_start.y +
      (float(yi) + 0.5f) / float(volume.size(2)) * (volume_end.y - volume_start.y);
    // Scan position on the wall
    const float3 scan_point = make_float3(
      scan_points[b][0][s],
      scan_points[b][1][s],
      scan_points[b][2][s]);
    // Laser position on the wall
    const float3 laser_point = make_float3(
      laser_points[b][0][l],
      laser_points[b][1][l],
      laser_points[b][2][l]);
    for (int zi=0; zi<volume.size(1); ++zi) {
      const float occupancy = volume[b][zi][yi][xi];
      if (occupancy == 0.f) {
        continue;
      }
      const float zf = volume_start.z +
        (float(zi) + 0.5f) / float(volume.size(1)) * (volume_end.z - volume_start.z);
      const float zf0 = zf - 0.5f / float(volume.size(1)) * (volume_end.z - volume_start.z);
      const float zf1 = zf + 0.5f / float(volume.size(1)) * (volume_end.z - volume_start.z);
      float d0 = distance(scan_point, make_float3(xf, yf, zf0)) + distance(laser_point, make_float3(xf, yf, zf0));
      float d1 = distance(scan_point, make_float3(xf, yf, zf1)) + distance(laser_point, make_float3(xf, yf, zf1));
      if (d1 < d0) {
        float t = d0;
        d0 = d1;
        d1 = t;
      }
      float bf0 = (d0 - bin_offset) / bin_width;
      float bf1 = (d1 - bin_offset) / bin_width;
      float range = bf1 - bf0;
      int first_bin = floorf(bf0);
      int last_bin = floorf(bf1);
      if (first_bin < 0) {
        if (last_bin < 0) {
          continue;
        }
        bf0 = 0.f;
        first_bin = 0;
      }
      if (last_bin >= transient.size(1)) {
        if (first_bin >= transient.size(1)) {
          continue;
        }
        bf1 = transient.size(1) - 1;
        last_bin = transient.size(1) - 1;
      }
      if (first_bin == last_bin) {
        atomicAdd(&transient[b][first_bin][s][l], occupancy);
      } else {
        const float fractional_occupancy = occupancy / range;
        atomicAdd(&transient[b][first_bin][s][l], (1.f - (bf0 - float(first_bin))) * fractional_occupancy);
        for (int bin=first_bin+1; bin<last_bin; ++bin) {
          atomicAdd(&transient[b][bin][s][l], fractional_occupancy);
        }
        atomicAdd(&transient[b][last_bin][s][l], (bf1 - float(last_bin)) * fractional_occupancy);
      }
    }
  }
}

void volume_render_exhaustive_forward_cuda(
    const at::Tensor volume, // [B, D, H, W]
    const at::Tensor scan_points, // [B, 3, S]
    const at::Tensor laser_points, // [B, 3, L]
    const at::Tensor scan_origin, // [B, 3, S]
    const at::Tensor laser_origin, // [B, 3, L]
    at::Tensor transient, // [B, N, S, L]
    const std::array<float, 3>& volume_start, const std::array<float, 3>& volume_end,
    const float bin_width, const float bin_offset) {
  // Conversions
  const float3 volume_start_f = make_float3(volume_start[0], volume_start[1], volume_start[2]);
  const float3 volume_end_f = make_float3(volume_end[0], volume_end[1], volume_end[2]);
  // Dims
  const dim3 threads(8, 8, 8);
  const dim3 blocks(
    (volume.size(3)*volume.size(2) + threads.x - 1) / threads.x,
    (scan_points.size(2)           + threads.y - 1) / threads.y,
    (laser_points.size(2)          + threads.z - 1) / threads.z);
  for (int b=0; b < transient.size(0); ++b) {
    AT_DISPATCH_FLOATING_TYPES(
      transient.type(),
      "volume_render_exhaustive_forward_kernel",
      ([&] {volume_render_exhaustive_forward_kernel<scalar_t><<<blocks, threads>>>(
        volume.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        scan_points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        laser_points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        scan_origin.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        laser_origin.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        transient.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        volume_start_f, volume_end_f, bin_width, bin_offset, b
      );
    }));
  }
}

template <typename scalar_t>
__global__ void volume_render_exhaustive_grad_volume_kernel(
    const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> volume, // [B, D, H, W]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> scan_points, // [B, 3, S]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> laser_points, // [B, 3, L]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> scan_origin, // [B, 3, S]
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> laser_origin, // [B, 3, L]
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> transient,  // [B, N, S, L]
    const float3 volume_start, const float3 volume_end,
    const float bin_width, const float bin_offset,
    const int b) {
  const int xi = blockIdx.x * blockDim.x + threadIdx.x;
  const int yi = blockIdx.y * blockDim.y + threadIdx.y;
  const int zi = blockIdx.z * blockDim.z + threadIdx.z;
  if (xi < volume.size(0) && yi < volume.size(1) && zi < volume.size(2)){
    
  }
}

void volume_render_exhaustive_grad_volume_cuda(
    const at::Tensor transient_grad, // [B, N, S, L]
    const at::Tensor scan_points, // [B, 3, S]
    const at::Tensor laser_points, // [B, 3, L]
    const at::Tensor scan_origin, // [B, 3, S]
    const at::Tensor laser_origin, // [B, 3, L]
    at::Tensor volume_grad, // [B, D, H, W]
    const std::array<float, 3>& volume_start, const std::array<float, 3>& volume_end,
    const float bin_width, const float bin_offset) {
  // Conversions
  const float3 volume_start_f = make_float3(volume_start[0], volume_start[1], volume_start[2]);
  const float3 volume_end_f = make_float3(volume_end[0], volume_end[1], volume_end[2]);
  // Dims
  const dim3 threads(8, 8, 8);
  const dim3 blocks(
    (volume_grad.size(3) + threads.x - 1) / threads.x,
    (volume_grad.size(2) + threads.y - 1) / threads.y,
    (volume_grad.size(1) + threads.z - 1) / threads.z);
  for (int b=0; b < transient_grad.size(0); ++b) {
    AT_DISPATCH_FLOATING_TYPES(
      transient_grad.type(),
      "volume_render_exhaustive_grad_volume_kernel",
      ([&] {volume_render_exhaustive_grad_volume_kernel<scalar_t><<<blocks, threads>>>(
        transient_grad.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        scan_points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        laser_points.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        scan_origin.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        laser_origin.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        volume_grad.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        volume_start_f, volume_end_f, bin_width, bin_offset, b
      );
    }));
  }
}

} // namespace render
} // namespace totri
