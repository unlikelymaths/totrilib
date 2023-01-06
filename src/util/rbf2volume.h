#ifndef TOTRI_UTIL_RBF2VOLUME
#define TOTRI_UTIL_RBF2VOLUME

#include <array>
#include <torch/extension.h>

namespace totri {
namespace util {

void rbf2volume_gaussian_forward(
  const at::Tensor rbf, // [B, N, 4+A]
  at::Tensor volume, // [B, 1+A, D, H, W]
  const std::array<float, 3>& volume_start,
  const std::array<float, 3>& volume_end);

void rbf2volume_gaussian_grad(
  const at::Tensor rbf, // [B, N, 4+A]
  const at::Tensor volume, // [B, 1+A, D, H, W]
  const at::Tensor volume_grad, // [B, 1+A, D, H, W]
  at::Tensor rbf_grad, // [B, N, 4+A]
  const std::array<float, 3>& volume_start,
  const std::array<float, 3>& volume_end);

} // namespace util
} // namespace totri

#endif // TOTRI_UTIL_RBF2VOLUME