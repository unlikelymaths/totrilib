#ifndef TOTRI_COMMON_TRIANGLE
#define TOTRI_COMMON_TRIANGLE

#include <cuda_runtime.h>

#include "common/math.h"

namespace totri {

struct vert3 {

  __host__ __device__
  vert3()
    : vert0(make_float3(0.f, 0.f, 0.f)),
      vert1(make_float3(0.f, 0.f, 0.f)),
      vert2(make_float3(0.f, 0.f, 0.f)) {}

  __host__ __device__
  vert3(const float3& vert0, const float3& vert1, const float3& vert2)
    : vert0(vert0), vert1(vert1), vert2(vert2) {}

  __host__ __device__ inline vert3& operator+=(const float& b) {
    vert0 += b;
    vert1 += b;
    vert2 += b;
    return *this;
  }

  __host__ __device__ inline vert3& operator+=(const float3& b) {
    vert0 += b;
    vert1 += b;
    vert2 += b;
    return *this;
  }

  __host__ __device__ inline vert3& operator+=(const vert3& b) {
    vert0 += b.vert0;
    vert1 += b.vert1;
    vert2 += b.vert2;
    return *this;
  }

  __host__ __device__ inline vert3& operator-=(const float& b) {
    vert0 -= b;
    vert1 -= b;
    vert2 -= b;
    return *this;
  }

  __host__ __device__ inline vert3& operator-=(const float3& b) {
    vert0 -= b;
    vert1 -= b;
    vert2 -= b;
    return *this;
  }

  __host__ __device__ inline vert3& operator-=(const vert3& b) {
    vert0 -= b.vert0;
    vert1 -= b.vert1;
    vert2 -= b.vert2;
    return *this;
  }

  __host__ __device__ inline vert3& operator*=(const float& b) {
    vert0 *= b;
    vert1 *= b;
    vert2 *= b;
    return *this;
  }

  __host__ __device__ inline vert3& operator/=(const float& b) {
    vert0 /= b;
    vert1 /= b;
    vert2 /= b;
    return *this;
  }

  float3 vert0;
  float3 vert1;
  float3 vert2;
};

__host__ __device__ inline vert3 operator+(const vert3& a, const vert3& b) {
  return vert3(a.vert0 + b.vert0, a.vert1 + b.vert1, a.vert2 + b.vert2);
}

__host__ __device__ inline vert3 operator+(const vert3& a, const float b) {
  return vert3(a.vert0 + b, a.vert1 + b, a.vert2 + b);
}

__host__ __device__ inline vert3 operator+(const float a, const vert3& b) {
  return vert3(a + b.vert0, a + b.vert1, a + b.vert2);
}

__host__ __device__ inline vert3 operator-(const vert3& a, const vert3& b) {
  return vert3(a.vert0 - b.vert0, a.vert1 - b.vert1, a.vert2 - b.vert2);
}

__host__ __device__ inline vert3 operator-(const vert3& a, const float b) {
  return vert3(a.vert0 - b, a.vert1 - b, a.vert2 - b);
}

__host__ __device__ inline vert3 operator-(const float a, const vert3& b) {
  return vert3(a - b.vert0, a - b.vert1, a - b.vert2);
}

__host__ __device__ inline vert3 operator*(const vert3& a, const float b) {
  return vert3(a.vert0 * b, a.vert1 * b, a.vert2 * b);
}

__host__ __device__ inline vert3 operator*(const float a, const vert3& b) {
  return vert3(a * b.vert0, a * b.vert1, a * b.vert2);
}

__host__ __device__ inline vert3 operator/(const vert3& a, const float b) {
  return vert3(a.vert0 / b, a.vert1 / b, a.vert2 / b);
}

__host__ __device__ inline vert3 operator/(const float a, const vert3& b) {
  return vert3(a / b.vert0, a / b.vert1, a / b.vert2);
}

} // namespace totri

#endif // TOTRI_COMMON_TRIANGLE
