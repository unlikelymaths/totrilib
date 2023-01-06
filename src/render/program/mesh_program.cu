/**
 * @file
 * @copydoc render/program/mesh_program.h
 */

#include "render/program/mesh_program.h"

namespace totri {
namespace render {

MeshProgramBase::MeshProgramBase(int device)
  : optix_context_(device),
    program_groups_(3) {}

OptixPipeline& MeshProgramBase::pipeline() {
  Init();
  return pipeline_;
}

OptixShaderBindingTable& MeshProgramBase::sbt() {
  Init();
  return sbt_;
}

bool MeshProgramBase::IsInitialized() const {
  return initialized_;
};

void MeshProgramBase::SetInitialized() {
  initialized_ = true;
}

void MeshProgramBase::InitPipeline(
    const std::string& pipelineLaunchParamsVariableName,
    const std::string& ptx_code) {
  // Create Module
  OptixModule module;
  OptixModuleCompileOptions moduleCompileOptions = {};
  moduleCompileOptions.maxRegisterCount          = 50;
  moduleCompileOptions.optLevel                  = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel                = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  OptixPipelineCompileOptions pipelineCompileOptions      = {};
  pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipelineCompileOptions.usesMotionBlur                   = false;
  pipelineCompileOptions.numPayloadValues                 = 1;
  pipelineCompileOptions.numAttributeValues               = 0;
  pipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
  pipelineCompileOptions.pipelineLaunchParamsVariableName = pipelineLaunchParamsVariableName.c_str();
    

  OPTIX_CHECK_AND_LOG(
    optixModuleCreateFromPTX(
      optix_context_->context(),
      &moduleCompileOptions,
      &pipelineCompileOptions,
      ptx_code.c_str(),
      ptx_code.size(),
      log,&sizeof_log,
      &module),
    #ifdef VERBOSE
    std::cout << logstring;
    #endif
  );

  // Create Programs
  OptixProgramGroupOptions pg_options = {};
  std::vector<OptixProgramGroupDesc> pg_descriptions(3);
  pg_descriptions[0].kind                         = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pg_descriptions[0].raygen.module                = module;           
  pg_descriptions[0].raygen.entryFunctionName     = "__raygen__rg";
  pg_descriptions[1].kind                         = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pg_descriptions[1].miss.module                  = module;           
  pg_descriptions[1].miss.entryFunctionName       = "__miss__ms";
  pg_descriptions[2].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pg_descriptions[2].hitgroup.moduleCH            = module;           
  pg_descriptions[2].hitgroup.entryFunctionNameCH = "__closesthit__ch";
  OPTIX_CHECK_AND_LOG(
    optixProgramGroupCreate(
      optix_context_->context(),
      pg_descriptions.data(),
      (int) pg_descriptions.size(),
      &pg_options,
      log,&sizeof_log,
      program_groups_.data()),
    #ifdef VERBOSE
    std::cout << logstring;
    #endif
  );

  // Create Pipeline
  OptixPipelineLinkOptions pipelineLinkOptions = {};
  pipelineLinkOptions.maxTraceDepth            = 2;
  OPTIX_CHECK_AND_LOG(
    optixPipelineCreate(
      optix_context_->context(),
      &pipelineCompileOptions,
      &pipelineLinkOptions,
      program_groups_.data(),
      (int) program_groups_.size(),
      log,&sizeof_log,
      &pipeline_),
    #ifdef VERBOSE
    std::cout << logstring;
    #endif
  );
}

} // namespace render
} // namespace totri
