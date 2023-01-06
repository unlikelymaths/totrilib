#ifndef TOTRI_COMMON_BUFFER
#define TOTRI_COMMON_BUFFER

#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace totri {

template <class T>
class HostDeviceBuffer {
 public:
  HostDeviceBuffer(int length=1)
    : data_device_(sizeof(T) * length), data_host_(sizeof(T) * length) {}
  T* DevicePtr() {
    return (T*) thrust::raw_pointer_cast(data_device_.data());
  }
  CUdeviceptr CuDevicePtr() {
    return (CUdeviceptr) thrust::raw_pointer_cast(data_device_.data());
  }
  T* operator->() {
    return (T*) thrust::raw_pointer_cast(data_host_.data());
  }
  T* HostPtr() {
    return (T*) thrust::raw_pointer_cast(data_host_.data());
  }
  void Download() {
    thrust::copy(data_device_.begin(), data_device_.end(), data_host_.begin());
  }
  void Upload() {
    thrust::copy(data_host_.begin(), data_host_.end(), data_device_.begin());
  }
  int Size() {
    return data_host_.size() / sizeof(T);
  }
  int SizeInBytes() {
    return data_host_.size();
  }

 private:
  thrust::device_vector<unsigned char> data_device_;
  thrust::host_vector<unsigned char> data_host_;
};

template <class T=unsigned char>
class DeviceBuffer {
 public:
  DeviceBuffer(int length=1) : data_(sizeof(T) * length) {}
  T* DevicePtr() {
    return (T*) thrust::raw_pointer_cast(data_.data());
  }
  CUdeviceptr CuDevicePtr() {
    return (CUdeviceptr) thrust::raw_pointer_cast(data_.data());
  }
  void Expand(int length) {
    ExpandInBytes(length * sizeof(T));
  }
  void ExpandInBytes(int size_in_bytes) {
    if (size_in_bytes > data_.size()) {
      ResizeInBytes(size_in_bytes);
    }
  }
  void Resize(int length) {
    ResizeInBytes(sizeof(T) * length);
  }
  void ResizeInBytes(int size_in_bytes) {
    data_ = thrust::device_vector<unsigned char>(size_in_bytes);
  }
  int Size() {
    return data_.size() / sizeof(T);
  }
  int SizeInBytes() {
    return data_.size();
  }

 private:
  thrust::device_vector<unsigned char> data_;
};

} // namespace totri

#endif // TOTRI_COMMON_BUFFER
