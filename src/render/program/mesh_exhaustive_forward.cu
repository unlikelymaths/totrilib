/**
 * @file
 * @copydoc render/program/mesh_exhaustive_forward.h
 */
#include "render/program/mesh_exhaustive_forward.h"

extern "C" char render_program_mesh_exhaustive_forward[];

namespace totri {
namespace render {

MeshExhaustiveForwardProgram::MeshExhaustiveForwardProgram(int device)
  : MeshProgramBase(device) {}

HostDeviceBuffer<MeshExhaustiveForwardLaunchParams>& MeshExhaustiveForwardProgram::launch_params() {
  Init();
  return launch_params_;
};

void MeshExhaustiveForwardProgram::Init() {
  if (IsInitialized()) {
    return;
  }
  InitPipeline(
    "mesh_exhaustive_forward_launch_params",
    render_program_mesh_exhaustive_forward);
  InitSbt(raygen_record_, miss_record_, hitgroup_record_);
  SetInitialized();
}

} // namespace render
} // namespace totri
