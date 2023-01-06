/**
 * @file
 * @copydoc common/optix.h
 */
#include "common/optix.h"

#include <optix_function_table_definition.h>
#include <thrust/device_vector.h>

#include "common/const.h"

namespace totri {
namespace render {

TriangleGas::TriangleGas(int device) : optix_context_(device) {}

template<typename index_t>
__global__ void FacesTensor2Buffer(
    const at::PackedTensorAccessor32<index_t, 3, at::RestrictPtrTraits> faces_tensor,
    uint3* faces_buffer, int b) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < faces_tensor.size(1)){
    const index_t f0 = faces_tensor[b][i][0];
    const index_t f1 = faces_tensor[b][i][1];
    const index_t f2 = faces_tensor[b][i][2];
    faces_buffer[i] = make_uint3(
      f0 >= 0 ? f0 : MAX_UINT,
      f1 >= 0 ? f1 : MAX_UINT,
      f2 >= 0 ? f2 : MAX_UINT);
  }
}

int TriangleGas::Update(
    const at::Tensor& verts, // [B, V, 3]
    const at::Tensor& faces, // [B, F, 3]
    int b) {
  // Check Inputs
  if (!verts.is_contiguous()) {
    throw std::runtime_error("Tensor 'verts' must be contiguous.");
  }
  verts_ptr_ = (CUdeviceptr) verts[b].data_ptr();
  // Copy data arranging indices as a series of uint3
  faces_.Expand(faces.size(1));
  const int threads = 1024;
  const int blocks = (faces.size(1) + threads - 1) / threads;
  AT_DISPATCH_INDEX_TYPES(
    faces.type(),
    "FacesTensor2Buffer",
    ([&] {FacesTensor2Buffer<index_t><<<blocks, threads>>>(
      faces.packed_accessor32<index_t, 3, torch::RestrictPtrTraits>(),
      faces_.DevicePtr(), b
    );
  }));
  // Count the Number of valid faces
  thrust::device_ptr<unsigned int> faces_device_ptr((unsigned int*) faces_.DevicePtr());
  const int num_faces_ = thrust::distance(
    faces_device_ptr,
    thrust::find(faces_device_ptr, faces_device_ptr + 3*faces.size(1), MAX_UINT)) / 3;
  // Build GAS from triangles and indices
  // Expand buffer sizes as needed
  const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
  OptixBuildInput triangle_input = {};
  triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  triangle_input.triangleArray.vertexBuffers = &verts_ptr_;
  triangle_input.triangleArray.numVertices = 3*verts.size(1);
  triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangle_input.triangleArray.vertexStrideInBytes = 3*sizeof(float);
  triangle_input.triangleArray.flags = triangle_input_flags;
  triangle_input.triangleArray.numSbtRecords= 1;
  triangle_input.triangleArray.indexBuffer = faces_.CuDevicePtr();
  triangle_input.triangleArray.numIndexTriplets = num_faces_;
  triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3; 
  triangle_input.triangleArray.indexStrideInBytes = 3*sizeof(int);
  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(
    optix_context_->context(),
    &accel_options,
    &triangle_input,
    1,
    &gas_buffer_sizes));
  tmp_buffer_.ExpandInBytes(gas_buffer_sizes.tempSizeInBytes);
  gas_buffer_.ExpandInBytes(gas_buffer_sizes.outputSizeInBytes);
  OPTIX_CHECK(optixAccelBuild(
    optix_context_->context(),
    0,
    &accel_options,
    &triangle_input,
    1,
    tmp_buffer_.CuDevicePtr(),
    gas_buffer_sizes.tempSizeInBytes,
    gas_buffer_.CuDevicePtr(),
    gas_buffer_sizes.outputSizeInBytes,
    &gas_handle_,
    nullptr,
    0));
  return num_faces_;
}

OptixTraversableHandle TriangleGas::handle() {
  return gas_handle_;
}

DeviceBuffer<uint3>& TriangleGas::faces_buffer() {
  return faces_;
}

int TriangleGas::num_faces() {
  return num_faces_;
}


} // namespace render
} // namespace totri
