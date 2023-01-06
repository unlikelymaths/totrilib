/**
 * @file
 * @brief Mesh Program Utility Functions
 */
#ifndef TOTRI_RENDER_PROGRAM_MESH_PROGRAM_H
#define TOTRI_RENDER_PROGRAM_MESH_PROGRAM_H

#include <string>

#include "common/optix.h"

namespace totri {
namespace render {

class MeshProgramBase {
 public:
  MeshProgramBase(const MeshProgramBase&) = delete;
  MeshProgramBase& operator=(const MeshProgramBase&) = delete;
  MeshProgramBase(int device);

  OptixPipeline& pipeline();
  OptixShaderBindingTable& sbt();

 protected:
  bool IsInitialized() const; 
  void SetInitialized(); 
  virtual void Init() = 0;
  void InitPipeline(
    const std::string& pipelineLaunchParamsVariableName,
    const std::string& ptx_code);
  template<typename T0, typename T1, typename T2>
  void InitSbt(
    HostDeviceBuffer<T0>& raygen_record,
    HostDeviceBuffer<T1>& miss_record,
    HostDeviceBuffer<T2>& hitgroup_record);

 private:
  bool initialized_ = false;
	SharedOptixContext optix_context_;
  OptixPipeline pipeline_;
  OptixShaderBindingTable sbt_;
  std::vector<OptixProgramGroup> program_groups_;
};

template<typename T0, typename T1, typename T2>
void MeshProgramBase::InitSbt(
    HostDeviceBuffer<T0>& raygen_record,
    HostDeviceBuffer<T1>& miss_record,
    HostDeviceBuffer<T2>& hitgroup_record) {
  sbt_ =  {};
  OPTIX_CHECK(optixSbtRecordPackHeader(program_groups_[0], raygen_record.HostPtr()));
  raygen_record.Upload();
  sbt_.raygenRecord = raygen_record.CuDevicePtr();
  OPTIX_CHECK(optixSbtRecordPackHeader(program_groups_[1], miss_record.HostPtr()));
  miss_record.Upload();
  sbt_.missRecordBase          = miss_record.CuDevicePtr();
  sbt_.missRecordStrideInBytes = miss_record.SizeInBytes();
  sbt_.missRecordCount         = 1;
  OPTIX_CHECK(optixSbtRecordPackHeader(program_groups_[2], hitgroup_record.HostPtr()));
  hitgroup_record.Upload();
  sbt_.hitgroupRecordBase          = hitgroup_record.CuDevicePtr();
  sbt_.hitgroupRecordStrideInBytes = hitgroup_record.SizeInBytes();
  sbt_.hitgroupRecordCount         = 1;
}

} // namespace render
} // namespace totri

#endif // TOTRI_RENDER_PROGRAM_MESH_PROGRAM_H
