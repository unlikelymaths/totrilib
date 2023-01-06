#ifndef TOTRI_COMMON_MATH
#define TOTRI_COMMON_MATH

#include <cuda_runtime.h>

namespace totri {

// Value

template<typename T>
__device__ inline T      powt(T,T);
template<>
__device__ inline float  powt(float v, float e) {return powf(v, e); }
template<>
__device__ inline double powt(double v, double e) {return pow(v, e); }

template<typename T>
__device__ inline T      sqrtt(T);
template<>
__device__ inline float  sqrtt(float v) {return sqrtf(v); }
template<>
__device__ inline double sqrtt(double v) {return sqrt(v); }

template<typename T>
__device__ inline T      expt(T);
template<>
__device__ inline float  expt(float v) {return expf(v); }
template<>
__device__ inline double expt(double v) {return exp(v); }

template<typename T>
__device__ inline T signt(T x) { 
	return x > 0 ? T(1) : (x == 0 ? T(0) : T(-1));
}

//  float3

__device__ inline float3 operator+(const float3& a) {
  return a;
}

__device__ inline float3 operator+(const float3& a, const float3& b) {
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ inline float3 operator+(const float3& a, const float& b) {
  return make_float3(a.x+b, a.y+b, a.z+b);
}

__device__ inline float3 operator+(const float& a, const float3& b) {
  return make_float3(a+b.x, a+b.y, a+b.z);
}

__device__ inline float3 operator-(const float3& a) {
  return make_float3(-a.x, -a.y, -a.z);
}

__device__ inline float3 operator-(const float3& a, const float3& b) {
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ inline float3 operator-(const float3& a, const float& b) {
  return make_float3(a.x-b, a.y-b, a.z-b);
}

__device__ inline float3 operator-(const float& a, const float3& b) {
  return make_float3(a-b.x, a-b.y, a-b.z);
}

__device__ inline float3 operator*(const float3& a, const float& b) {
  return make_float3(a.x*b, a.y*b, a.z*b);
}

__device__ inline float3 operator*(const float& a, const float3& b) {
  return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__ inline float3 operator/(const float3& a, const float& b) {
  return make_float3(a.x/b, a.y/b, a.z/b);
}

__device__ inline float3 operator/(const float& a, const float3& b) {
  return make_float3(a/b.x, a/b.y, a/b.z);
}

__device__ inline float3& operator+=(float3& a, const float b) {
  a.x += b;
  a.y += b;
  a.z += b;
  return a;
}

__device__ inline float3& operator+=(float3& a, const float3& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

__device__ inline float3& operator-=(float3& a, const float b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  return a;
}

__device__ inline float3& operator-=(float3& a, const float3& b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

__device__ inline float3& operator*=(float3& a, const float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  return a;
}

__device__ inline float3& operator/=(float3& a, const float b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  return a;
}

__device__ inline float dot(const float3& a, const float3& b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ inline float distance(const float3& a, const float3& b) {
  return norm3df(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ inline float length(const float3& a) {
  return norm3df(a.x, a.y, a.z);
}

__device__ inline float length_sqr(const float3& a) {
  return a.x*a.x + a.y*a.y + a.z*a.z;
}

__device__ inline float3 normalized(const float3& a) {
  return a / length(a);
}

__device__ inline float3 cross(const float3& a, const float3& b) {
  return make_float3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );
}

__device__ inline void swap(int& a, int& b) {
  const int t = a;
  a = b;
  b = t;
}

__device__ inline void swap(unsigned int& a, unsigned int& b) {
  const unsigned int t = a;
  a = b;
  b = t;
}

__device__ inline void swap(float& a, float& b) {
  const float t = a;
  a = b;
  b = t;
}

__device__ inline void swap(float3& a, float3& b) {
  swap(a.x, b.x);
  swap(a.y, b.y);
  swap(a.z, b.z);
}

// float4

__device__ inline float4 operator*(const float& a, const float4& b) {
  return make_float4(a*b.x, a*b.y, a*b.z, a*b.w);
}

__device__ inline float4 operator*(const float4& a, const float& b) {
  return make_float4(a.x*b, a.y*b, a.z*b, a.w*b);
}

} // namespace totri

#endif // TOTRI_COMMON_MATH
