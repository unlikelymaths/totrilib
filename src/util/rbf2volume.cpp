#include "util/rbf2volume.h"
#include "util/rbf2volume_cuda.h"

namespace totri {
namespace util {

void rbf2volume_gaussian_forward(
    const at::Tensor rbf, // [B, N, 4+A]
    at::Tensor volume, // [B, 1+A, D, H, W]
    const std::array<float, 3>& volume_start,
    const std::array<float, 3>& volume_end) {
  rbf2volume_gaussian_forward_cuda(
    rbf, volume,
    volume_start, volume_end);
}

void rbf2volume_gaussian_grad(
    const at::Tensor rbf, // [B, N, 4+A]
    const at::Tensor volume, // [B, 1+A, D, H, W]
    const at::Tensor volume_grad, // [B, 1+A, D, H, W]
    at::Tensor rbf_grad, // [B, N, 4+A]
    const std::array<float, 3>& volume_start,
    const std::array<float, 3>& volume_end) {
  rbf2volume_gaussian_grad_cuda(
    rbf, volume, volume_grad, rbf_grad,
    volume_start, volume_end);
}

} // namespace util
} // namespace totri
