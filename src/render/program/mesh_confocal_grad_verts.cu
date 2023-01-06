/**
 * @file
 * @copydoc render/program/mesh_confocal_grad_verts.h
 */
#include "render/program/mesh_confocal_grad_verts.h"

extern "C" char render_program_mesh_confocal_grad_verts[];

namespace totri {
namespace render {

MeshConfocalGradVertsProgram::MeshConfocalGradVertsProgram(int device)
  : MeshProgramBase(device) {}

HostDeviceBuffer<MeshConfocalGradVertsLaunchParams>& MeshConfocalGradVertsProgram::launch_params() {
  Init();
  return launch_params_;
};

void MeshConfocalGradVertsProgram::Init() {
  if (IsInitialized()) {
    return;
  }
  InitPipeline(
    "mesh_confocal_grad_verts_launch_params",
    render_program_mesh_confocal_grad_verts);
  InitSbt(raygen_record_, miss_record_, hitgroup_record_);
  SetInitialized();
}

} // namespace render
} // namespace totri
