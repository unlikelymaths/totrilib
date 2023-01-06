cmake_minimum_required(VERSION 3.18)
project(totri_cpp LANGUAGES CUDA CXX)

# General
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Pybind11
add_subdirectory(extern/pybind11)

# Torch
if(NOT DEFINED Torch_ROOT)
  message(WARNING "Torch_ROOT not set. Trying to find Torch anyway.")
endif()
find_package(Torch REQUIRED)
find_library(Torch_PYTHON_LIB torch_python PATHS "${Torch_ROOT}/lib")
set(TORCH_LIBRARIES "${TORCH_LIBRARIES};${Torch_PYTHON_LIB}")

# Optix
if(NOT DEFINED ENV{OptiX_INSTALL_DIR})
  message(WARNING "OptiX_INSTALL_DIR not set. Trying to find Optix anyway.")
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/configure_optix.cmake)

# Totri
string(
  REPLACE ";" ","
  OPTIX_PROGRAM_INCLUDES
  "${CMAKE_CURRENT_SOURCE_DIR}/src;${TORCH_INCLUDE_DIRS};${PYTHON_INCLUDE_DIRS}"
)
message("OPTIX_PROGRAM_INCLUDES: ${OPTIX_PROGRAM_INCLUDES}")
cuda_compile_and_embed(render_program_mesh_confocal_forward
  src/render/program/mesh_confocal_forward_optix.cu
  ${OPTIX_PROGRAM_INCLUDES}
)
cuda_compile_and_embed(render_program_mesh_confocal_grad_verts
  src/render/program/mesh_confocal_grad_verts_optix.cu
  ${OPTIX_PROGRAM_INCLUDES}
)
cuda_compile_and_embed(render_program_mesh_exhaustive_forward
  src/render/program/mesh_exhaustive_forward_optix.cu
  ${OPTIX_PROGRAM_INCLUDES}
)
cuda_compile_and_embed(render_program_mesh_exhaustive_grad_verts
  src/render/program/mesh_exhaustive_grad_verts_optix.cu
  ${OPTIX_PROGRAM_INCLUDES}
)
set(OPTIX_GEN_SRC
  ${render_program_mesh_confocal_forward}
  ${render_program_mesh_confocal_grad_verts}
  ${render_program_mesh_exhaustive_forward}
  ${render_program_mesh_exhaustive_grad_verts}
)
set(CPP_SRC
  src/reconstruct/backprojection.cpp
  src/render/mesh.cpp
  src/render/volume.cpp
  src/util/rbf2volume.cpp
  src/totri.cpp
)
set(CUDA_SRC
  src/common/optix.cu
  src/reconstruct/backprojection_cuda.cu
  src/render/program/mesh_confocal_forward.cu
  src/render/program/mesh_confocal_grad_verts.cu
  src/render/program/mesh_exhaustive_forward.cu
  src/render/program/mesh_exhaustive_grad_verts.cu
  src/render/program/mesh_program.cu
  src/render/mesh_cuda.cu
  src/render/volume_cuda.cu
  src/util/rbf2volume_cuda.cu
)

pybind11_add_module(totri_cpp ${CPP_SRC} ${CUDA_SRC} ${OPTIX_GEN_SRC})
set_target_properties(totri_cpp PROPERTIES CUDA_ARCHITECTURES "61;75;86")
target_include_directories(totri_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(totri_cpp PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(totri_cpp PUBLIC "${TORCH_LIBRARIES}")
target_compile_definitions(totri_cpp PRIVATE VERSION_INFO=${VERSION_INFO})
