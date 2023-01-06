#include "util/rbf2volume_cuda.h"
#include <cuda_runtime.h>

#include "common/math.h"

namespace totri {
namespace util {

#define NTHREADS_X 4
#define NTHREADS_Y 4
#define NTHREADS_Z 4
#define NTHREADS 64
#define MAX_ATTR 8

template <typename scalar_t>
__device__ scalar_t Gaussian(
    const scalar_t x,  const scalar_t y,  const scalar_t z,
    const scalar_t xr, const scalar_t yr, const scalar_t zr,
    const scalar_t sigma) {
  return expt(-0.5f *
    (
      (xr - x) * (xr - x) +
      (yr - y) * (yr - y) +
      (zr - z) * (zr - z)
    ) / (
      sigma * sigma
    )
  );
}

template <typename scalar_t>
__device__ void GaussianGradParams(
    const scalar_t x,  const scalar_t y,  const scalar_t z,
    const scalar_t xr, const scalar_t yr, const scalar_t zr,
    const scalar_t sigma, const scalar_t weight,
    scalar_t& dweight_dxr, scalar_t& dweight_dyr,
    scalar_t& dweight_dzr, scalar_t& dweight_dsigmar) {
  const scalar_t sigmainv = scalar_t(1) / sigma;
  dweight_dxr = (x - xr) * sigmainv * sigmainv * weight;
  dweight_dyr = (y - yr) * sigmainv * sigmainv * weight;
  dweight_dzr = (z - zr) * sigmainv * sigmainv * weight;
  dweight_dsigmar = (
      (xr - x) * (xr - x) +
      (yr - y) * (yr - y) +
      (zr - z) * (zr - z)
    ) * sigmainv * sigmainv * sigmainv * weight;
}

template <typename scalar_t>
__global__ void rbf2volume_gaussian_forward_kernel(
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> rbf, // [B, N, 4+A]
    at::PackedTensorAccessor32<scalar_t, 5, at::RestrictPtrTraits> volume, // [B, 1+A, D, H, W]
    const float3 volume_start, const float3 volume_end, int b) {
  // Spatial position
  const int xi = blockIdx.x * blockDim.x + threadIdx.x;
  const int yi = blockIdx.y * blockDim.y + threadIdx.y;
  const int zi = blockIdx.z * blockDim.z + threadIdx.z;
  const scalar_t x = (volume.size(4) == 1) ? scalar_t(0.5)*scalar_t(volume_start.x + volume_end.x) : scalar_t(volume_start.x) + scalar_t(volume_end.x - volume_start.x) * scalar_t(xi) / scalar_t(volume.size(4) - 1);
  const scalar_t y = (volume.size(3) == 1) ? scalar_t(0.5)*scalar_t(volume_start.y + volume_end.y) : scalar_t(volume_start.y) + scalar_t(volume_end.y - volume_start.y) * scalar_t(yi) / scalar_t(volume.size(3) - 1);
  const scalar_t z = (volume.size(2) == 1) ? scalar_t(0.5)*scalar_t(volume_start.z + volume_end.z) : scalar_t(volume_start.z) + scalar_t(volume_end.z - volume_start.z) * scalar_t(zi) / scalar_t(volume.size(2) - 1);
  const int idx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  // Attribues
  const int num_attr = rbf.size(2) - 4;
  __shared__ scalar_t attr_values[MAX_ATTR * NTHREADS];
  for (int a=0; a < num_attr; ++a) {
      attr_values[a * NTHREADS + idx] = scalar_t(0);
  }
  __syncthreads();
  if (xi < volume.size(4) && yi < volume.size(3) && zi < volume.size(2)) {
    // Sum basis functions
    scalar_t sum = scalar_t(0);
    for(int i=0; i<rbf.size(1); ++i) {
      const scalar_t xr = rbf[b][i][0];
      const scalar_t yr = rbf[b][i][1];
      const scalar_t zr = rbf[b][i][2];
      const scalar_t sigma = rbf[b][i][3];
      const scalar_t weight = Gaussian(x, y, z, xr, yr, zr, sigma);
      for (int a=0; a < num_attr; ++a) {
        attr_values[a * NTHREADS + idx] += weight * rbf[b][i][4+a];
      }
      sum += weight;
    }
    volume[b][0][zi][yi][xi] = sum;
    // To avoid instabilities for small weights
    if (sum < scalar_t(1.e-8)) {
      sum = scalar_t(1);
    }
    // Write normalized attribute values
    for (int a=0; a < num_attr; ++a) {
      volume[b][1+a][zi][yi][xi] = attr_values[a * NTHREADS + idx] / sum;
    }
  }
}

void rbf2volume_gaussian_forward_cuda(
    const at::Tensor rbf, // [B, N, 4+A]
    at::Tensor volume, // [B, 1+A, D, H, W]
    const std::array<float, 3>& volume_start,
    const std::array<float, 3>& volume_end) {
  // Conversions
  const float3 volume_start_f = make_float3(volume_start[0], volume_start[1], volume_start[2]);
  const float3 volume_end_f = make_float3(volume_end[0], volume_end[1], volume_end[2]);
  // Dims
  const dim3 threads(NTHREADS_X, NTHREADS_Y, NTHREADS_Z);
  const dim3 blocks(
    (volume.size(4) + threads.x - 1) / threads.x,
    (volume.size(3) + threads.y - 1) / threads.y,
    (volume.size(2) + threads.z - 1) / threads.z);
  for (int b=0; b < rbf.size(0); ++b) {
    AT_DISPATCH_FLOATING_TYPES(
      rbf.type(),
      "rbf2volume_gaussian_forward_kernel",
      ([&] {rbf2volume_gaussian_forward_kernel<scalar_t><<<blocks, threads>>>(
        rbf.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        volume.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
        volume_start_f, volume_end_f, b
      );
    }));
  }
}


template <typename scalar_t>
__global__ void rbf2volume_gaussian_grad_kernel(
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> rbf, // [B, N, 4+A]
    const at::PackedTensorAccessor32<scalar_t, 5, at::RestrictPtrTraits> volume, // [B, 1+A, D, H, W]
    const at::PackedTensorAccessor32<scalar_t, 5, at::RestrictPtrTraits> volume_grad, // [B, 1+A, D, H, W]
    at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> rbf_grad, // [B, N, 4+A]
    const float3 volume_start, const float3 volume_end, int b) {
  // Spatial position
  const int xi = (blockIdx.x * blockDim.x + threadIdx.x) % volume.size(4);
  const int yi = (blockIdx.x * blockDim.x + threadIdx.x) / volume.size(4);
  const int zi = blockIdx.y * blockDim.y + threadIdx.y;
  const int ri = blockIdx.z * blockDim.z + threadIdx.z;
  const scalar_t x = (volume.size(4) == 1) ? scalar_t(0.5)*scalar_t(volume_start.x + volume_end.x) : scalar_t(volume_start.x) + scalar_t(volume_end.x - volume_start.x) * scalar_t(xi) / scalar_t(volume.size(4) - 1);
  const scalar_t y = (volume.size(3) == 1) ? scalar_t(0.5)*scalar_t(volume_start.y + volume_end.y) : scalar_t(volume_start.y) + scalar_t(volume_end.y - volume_start.y) * scalar_t(yi) / scalar_t(volume.size(3) - 1);
  const scalar_t z = (volume.size(2) == 1) ? scalar_t(0.5)*scalar_t(volume_start.z + volume_end.z) : scalar_t(volume_start.z) + scalar_t(volume_end.z - volume_start.z) * scalar_t(zi) / scalar_t(volume.size(2) - 1);
  const int idx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int num_attr = rbf.size(2) - 4;
  // Compute Grad
  __shared__ scalar_t dloss_drbf[(4 + MAX_ATTR) * NTHREADS];
  for (int a=0; a < rbf.size(2); ++a) {
      dloss_drbf[a * NTHREADS + idx] = scalar_t(0);
  }
  if (xi < volume.size(4) && yi < volume.size(3) && zi < volume.size(2)) {
    const scalar_t xr = rbf[b][ri][0];
    const scalar_t yr = rbf[b][ri][1];
    const scalar_t zr = rbf[b][ri][2];
    const scalar_t sigma = rbf[b][ri][3];
    const scalar_t weight = Gaussian(x, y, z, xr, yr, zr, sigma);
    const scalar_t dloss_dvolume = volume_grad[b][0][zi][yi][xi];
    scalar_t dweight_dxr;
    scalar_t dweight_dyr;
    scalar_t dweight_dzr;
    scalar_t dweight_dsigmar;
    GaussianGradParams(
      x, y, z, xr, yr, zr, sigma, weight,
      dweight_dxr, dweight_dyr, dweight_dzr, dweight_dsigmar);
    if (num_attr == 0) {
      dloss_drbf[0 * NTHREADS + idx] = dloss_dvolume * dweight_dxr;
      dloss_drbf[1 * NTHREADS + idx] = dloss_dvolume * dweight_dyr;
      dloss_drbf[2 * NTHREADS + idx] = dloss_dvolume * dweight_dzr;
      dloss_drbf[3 * NTHREADS + idx] = dloss_dvolume * dweight_dsigmar;
    } else {
      scalar_t volume_val = volume[b][0][zi][yi][xi];
      // To avoid instabilities for small weights
      if (volume_val < scalar_t(1.e-8)) {
        volume_val = scalar_t(1);
      }
      const scalar_t volume_inv = scalar_t(1) / volume_val;
      scalar_t sum = scalar_t(0);
      for (int a=0; a < num_attr; ++a) {
        const scalar_t attr = volume[b][1+a][zi][yi][xi];
        const scalar_t dloss_dattr = volume_grad[b][1+a][zi][yi][xi];
        // drbf
        sum += dloss_dattr * (rbf[b][ri][4+a] - attr);
        // dattr
        dloss_drbf[(4 + a) * NTHREADS + idx] = dloss_dattr * weight * volume_inv;
      }
      sum = sum * volume_inv;
      dloss_drbf[0 * NTHREADS + idx] = dweight_dxr     * (dloss_dvolume + sum);
      dloss_drbf[1 * NTHREADS + idx] = dweight_dyr     * (dloss_dvolume + sum);
      dloss_drbf[2 * NTHREADS + idx] = dweight_dzr     * (dloss_dvolume + sum);
      dloss_drbf[3 * NTHREADS + idx] = dweight_dsigmar * (dloss_dvolume + sum);
    }
  }
  // Sum gradients over voxels
  __syncthreads();
  for (int off=1; off<NTHREADS; off*=2) {
    if (idx % (2*off) == 0 && idx + off < NTHREADS) {
      for (int a=0; a < rbf.size(2); ++a) {
        dloss_drbf[a * NTHREADS + idx] += dloss_drbf[a * NTHREADS + idx + off];
      }
    }
    __syncthreads();
  }
  // Write output
  if (idx < rbf.size(2)) {
    atomicAdd(&rbf_grad[b][ri][idx], dloss_drbf[idx * NTHREADS]);
  }
}

void rbf2volume_gaussian_grad_cuda(
    const at::Tensor rbf, // [B, N, 4+A]
    const at::Tensor volume, // [B, 1+A, D, H, W]
    const at::Tensor volume_grad, // [B, 1+A, D, H, W]
    at::Tensor rbf_grad, // [B, N, 4+A]
    const std::array<float, 3>& volume_start,
    const std::array<float, 3>& volume_end) {
  // Conversions
  const float3 volume_start_f = make_float3(volume_start[0], volume_start[1], volume_start[2]);
  const float3 volume_end_f = make_float3(volume_end[0], volume_end[1], volume_end[2]);
  // Dims
  const dim3 threads(NTHREADS_X * NTHREADS_Y, NTHREADS_Z, 1);
  const dim3 blocks(
    (volume.size(4)*volume.size(3) + threads.x - 1) / threads.x,
    (volume.size(2) + threads.y - 1) / threads.y,
    rbf.size(1));
  for (int b=0; b < rbf.size(0); ++b) {
    AT_DISPATCH_FLOATING_TYPES(
      rbf.type(),
      "rbf2volume_gaussian_grad_kernel",
      ([&] {rbf2volume_gaussian_grad_kernel<scalar_t><<<blocks, threads>>>(
        rbf.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        volume.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
        volume_grad.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
        rbf_grad.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
        volume_start_f, volume_end_f, b
      );
    }));
  }
}

} // namespace util
} // namespace totri
