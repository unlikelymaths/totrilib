#ifndef TOTRI_COMMON_OPTIX
#define TOTRI_COMMON_OPTIX

#include <map>
#include <memory>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <torch/extension.h>

#include "common/buffer.h"

#define OPTIX_CHECK(cmd) { \
  char log[2048]; \
  size_t sizeof_log = sizeof(log); \
  OptixResult optix_result = cmd; \
  if (optix_result != OPTIX_SUCCESS) { \
    throw totri::render::OptixError(optix_result, \
      std::string("In file ") + std::string(__FILE__) + \
      std::string(", line ") + std::to_string(__LINE__) + \
      std::string(": ") + std::string(log, sizeof_log)); \
  } \
}

#define OPTIX_CHECK_AND_LOG(cmd, logcmd) { \
  char log[2048]; \
  size_t sizeof_log = sizeof(log); \
  OptixResult optix_result = cmd; \
  if (optix_result != OPTIX_SUCCESS) { \
    throw totri::render::OptixError(optix_result, \
      std::string("In file ") + std::string(__FILE__) + \
      std::string("(") + std::to_string(__LINE__) + \
      std::string("): ") + std::string(log, sizeof_log)); \
  } \
  if (sizeof_log > 1) { \
    std::string logstring(log, sizeof_log); \
    logcmd; \
  } \
}

namespace totri {
namespace render {

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) EmptyRaygenRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) EmptyMissRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) EmptyHitgroupRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

class OptixError : public std::runtime_error
{
 public:
  OptixError(const OptixResult& optix_result, const std::string& msg)
    : std::runtime_error(ErrorString(optix_result, msg)) {}

 private:
  static std::string ErrorString(const OptixResult& optix_result,
                                 const std::string& msg) {
    std::string error_message =
      std::string("Optix Error ") +
      std::string(optixGetErrorName(optix_result)) +
      std::string(": ") +
      std::string(optixGetErrorString(optix_result));
    if (msg.size() > 0) {
      error_message += std::string(" (") + msg + std::string(")");
    }
    return error_message;
  }
};

class OptixContext {
 public:
  static std::shared_ptr<OptixContext> Get(int device) {
    static std::map<int, std::weak_ptr<OptixContext>> contexts;
    auto it = contexts.find(device);
    if (it != contexts.end()) {
      if (!it->second.expired())
        return it->second.lock();
    }
    std::shared_ptr<OptixContext> new_context = std::shared_ptr<OptixContext>(
      new OptixContext(device));
    contexts[device] = new_context;
    return new_context;
  }

  const OptixDeviceContext& context() { return context_; }

  int device() { return device_; }

 private:
  OptixContext(int device) : device_(device) {
    cudaSetDevice(device);
    cudaFree(0);
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    CUcontext cuCtx = 0;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context_));
  }

  OptixDeviceContext context_;
  const int device_;
};

class SharedOptixContext : public std::shared_ptr<OptixContext> {
 public:
  SharedOptixContext(int device) : std::shared_ptr<OptixContext>(OptixContext::Get(device)) {}
};

class TriangleGas {
 public:
  TriangleGas(int device);
  int Update(
    const at::Tensor& verts, // [B, V, 3]
    const at::Tensor& faces, // [B, F, 3]
    int b);

  OptixTraversableHandle handle();
  DeviceBuffer<uint3>& faces_buffer();
  int num_faces();

 private:
	SharedOptixContext optix_context_;
  CUdeviceptr verts_ptr_;
	DeviceBuffer<uint3>  faces_;
  OptixTraversableHandle gas_handle_;
  DeviceBuffer<> gas_buffer_;
	DeviceBuffer<> tmp_buffer_;
  int num_faces_;
};

} // namespace render
} // namespace totri

#endif // TOTRI_COMMON_OPTIX
