/**
 * @file
 * @copydoc render/program/mesh_confocal_forward.h
 */
#include "render/program/mesh_confocal_forward.h"

extern "C" char render_program_mesh_confocal_forward[];

namespace totri {
namespace render {

MeshConfocalForwardProgram::MeshConfocalForwardProgram(int device)
  : MeshProgramBase(device) {}

HostDeviceBuffer<MeshConfocalForwardLaunchParams>& MeshConfocalForwardProgram::launch_params() {
  Init();
  return launch_params_;
};

void MeshConfocalForwardProgram::Init() {
  if (IsInitialized()) {
    return;
  }
  InitPipeline(
    "mesh_confocal_forward_launch_params",
    render_program_mesh_confocal_forward);
  InitSbt(raygen_record_, miss_record_, hitgroup_record_);
  SetInitialized();
}

} // namespace render
} // namespace totri
