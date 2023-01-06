#include "render/mesh_cuda.h"

#include <optix.h>
#include <optix_stubs.h>
#include <thrust/device_vector.h>

#include "common/const.h"

namespace totri {
namespace render {

MeshConfocalContext::MeshConfocalContext(int device)
    : optix_context_(device),
      gas_(device),
      forward_program_(device),
      grad_verts_program_(device) {}

void MeshConfocalContext::Forward(
    const at::Tensor verts, // [B, V, 3]
    const at::Tensor faces, // [B, F, 3]
    const at::Tensor material, // [B, V, M]
    const at::Tensor scan_points, // [B, S, 3]
    const at::Tensor scan_origin, // [B, S, 3]
    at::Tensor transient, // [B, T, S]
    const float bin_width, const float bin_offset, const int model) {
  // Reset
  transient.fill_(0.f);
  if (faces.size(1) == 0) {
    return;
  }
  // Iterate over batch
  for (int b=0; b<verts.size(0); ++b) {
    int num_faces = gas_.Update(verts, faces, b);
    // Render transient image
    HostDeviceBuffer<MeshConfocalForwardLaunchParams>& launch_params = forward_program_.launch_params();
    launch_params->handle      = gas_.handle();
    launch_params->verts       = verts.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->material    = material.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->scan_points = scan_points.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->scan_origin = scan_origin.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->transient   = transient.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->faces       = gas_.faces_buffer().DevicePtr();
    launch_params->bin_width   = bin_width;
    launch_params->bin_offset  = bin_offset;
    launch_params->model       = model;
    launch_params->b           = b;
    launch_params.Upload();
    OPTIX_CHECK(optixLaunch(
      forward_program_.pipeline(), 0,
      launch_params.CuDevicePtr(),
      launch_params.SizeInBytes(),
      &forward_program_.sbt(),
      num_faces,
      scan_points.size(1),
      1
      ));
  }
}

void MeshConfocalContext::GradVerts(
    const at::Tensor verts, // [B, V, 3]
    const at::Tensor faces, // [B, F, 3]
    const at::Tensor material, // [B, V, M]
    const at::Tensor scan_points, // [B, S, 3]
    const at::Tensor scan_origin, // [B, S, 3]
    const at::Tensor transient_grad, // [B, T, S]
    at::Tensor verts_grad, // [B, V, 3]
    at::Tensor material_grad, // [B, V, M]
    const float bin_width, const float bin_offset, const int model) {
  // Reset
  verts_grad.fill_(0.f);
  material_grad.fill_(0.f);
  if (faces.size(1) == 0) {
    return;
  }
  // Iterate over batch
  for (int b=0; b<verts.size(0); ++b) {
    int num_faces = gas_.Update(verts, faces, b);
    // Render transient image
    HostDeviceBuffer<MeshConfocalGradVertsLaunchParams>& launch_params = grad_verts_program_.launch_params();
    launch_params->handle         = gas_.handle();
    launch_params->verts          = verts.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->material       = material.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->scan_points    = scan_points.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->scan_origin    = scan_origin.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->transient_grad = transient_grad.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->verts_grad     = verts_grad.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->material_grad  = material_grad.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->faces          = gas_.faces_buffer().DevicePtr();
    launch_params->bin_width      = bin_width;
    launch_params->bin_offset     = bin_offset;
    launch_params->model          = model;
    launch_params->b              = b;
    launch_params.Upload();
    OPTIX_CHECK(optixLaunch(
      grad_verts_program_.pipeline(), 0,
      launch_params.CuDevicePtr(),
      launch_params.SizeInBytes(),
      &grad_verts_program_.sbt(),
      num_faces,
      scan_points.size(1),
      1
      ));
  }
}

MeshExhaustiveContext::MeshExhaustiveContext(int device)
    : optix_context_(device),
      gas_(device),
      forward_program_(device),
      grad_verts_program_(device) {}

void MeshExhaustiveContext::Forward(
    const at::Tensor verts, // [B, V, 3]
    const at::Tensor faces, // [B, F, 3]
    const at::Tensor material, // [B, V, M]
    const at::Tensor scan_points, // [B, S, 3]
    const at::Tensor laser_points, // [B, L, 3]
    const at::Tensor scan_origin, // [B, S, 3]
    const at::Tensor laser_origin, // [B, L, 3]
    at::Tensor transient, // [B, T, S, L]
    const float bin_width, const float bin_offset, const int model) {
  // Reset
  transient.fill_(0.f);
  if (faces.size(1) == 0) {
    return;
  }
  // Iterate over batch
  for (int b=0; b<verts.size(0); ++b) {
    int num_faces = gas_.Update(verts, faces, b);
    // Render transient image
    HostDeviceBuffer<MeshExhaustiveForwardLaunchParams>& launch_params = forward_program_.launch_params();
    launch_params->handle       = gas_.handle();
    launch_params->verts        = verts.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->material     = material.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->scan_points  = scan_points.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->laser_points = laser_points.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->scan_origin  = scan_origin.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->laser_origin = laser_origin.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->transient    = transient.packed_accessor32<float, 4, torch::DefaultPtrTraits>();
    launch_params->faces        = gas_.faces_buffer().DevicePtr();
    launch_params->bin_width    = bin_width;
    launch_params->bin_offset   = bin_offset;
    launch_params->model        = model;
    launch_params->b            = b;
    launch_params.Upload();
    OPTIX_CHECK(optixLaunch(
      forward_program_.pipeline(), 0,
      launch_params.CuDevicePtr(),
      launch_params.SizeInBytes(),
      &forward_program_.sbt(),
      num_faces,
      scan_points.size(1),
      laser_points.size(1)
      ));
  }
}

void MeshExhaustiveContext::GradVerts(
    const at::Tensor verts, // [B, V, 3]
    const at::Tensor faces, // [B, F, 3]
    const at::Tensor material, // [B, V, M]
    const at::Tensor scan_points, // [B, S, 3]
    const at::Tensor laser_points, // [B, L, 3]
    const at::Tensor scan_origin, // [B, S, 3]
    const at::Tensor laser_origin, // [B, S, 3]
    const at::Tensor transient_grad, // [B, T, S, L]
    at::Tensor verts_grad, // [B, V, 3]
    at::Tensor material_grad, // [B, V, M]
    const float bin_width, const float bin_offset,
    const int model) {
  // Reset
  verts_grad.fill_(0.f);
  material_grad.fill_(0.f);
  if (faces.size(1) == 0) {
    return;
  }
  // Iterate over batch
  for (int b=0; b<verts.size(0); ++b) {
    int num_faces = gas_.Update(verts, faces, b);
    // Render transient image
    HostDeviceBuffer<MeshExhaustiveGradVertsLaunchParams>& launch_params = grad_verts_program_.launch_params();
    launch_params->handle         = gas_.handle();
    launch_params->verts          = verts.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->material       = material.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->scan_points    = scan_points.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->laser_points   = laser_points.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->scan_origin    = scan_origin.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->laser_origin   = laser_origin.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->transient_grad = transient_grad.packed_accessor32<float, 4, torch::DefaultPtrTraits>();
    launch_params->verts_grad     = verts_grad.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->material_grad  = material_grad.packed_accessor32<float, 3, torch::DefaultPtrTraits>();
    launch_params->faces          = gas_.faces_buffer().DevicePtr();
    launch_params->bin_width      = bin_width;
    launch_params->bin_offset     = bin_offset;
    launch_params->model          = model;
    launch_params->b              = b;
    launch_params.Upload();
    OPTIX_CHECK(optixLaunch(
      grad_verts_program_.pipeline(), 0,
      launch_params.CuDevicePtr(),
      launch_params.SizeInBytes(),
      &grad_verts_program_.sbt(),
      num_faces,
      scan_points.size(1),
      laser_points.size(1)
      ));
  }
}

} // namespace render
} // namespace totri
