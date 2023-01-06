#include "common/optix.h"
#include "reconstruct/backprojection.h"
#include "render/mesh.h"
#include "render/volume.h"
#include "util/rbf2volume.h"

#ifndef VERSION_INFO
#define VERSION_INFO dev
#endif

namespace totri {

PYBIND11_MODULE(totri_cpp, m) {
  m.attr("__version__") = "VERSION_INFO";

  py::register_exception<render::OptixError>(m, "OptixError");

  // reconstruct
  py::module reconstruct = m.def_submodule("reconstruct");
  reconstruct.def("backprojection_exhaustive_arellano_forward",
                  &reconstruct::backprojection_exhaustive_arellano_forward);
  reconstruct.def("backprojection_confocal_velten_forward",
                  &reconstruct::backprojection_confocal_velten_forward);
  reconstruct.def("backprojection_exhaustive_velten_forward",
                  &reconstruct::backprojection_exhaustive_velten_forward);

  // render
  py::module render = m.def_submodule("render");
  render.def("mesh_confocal_forward",
             &render::mesh_confocal_forward);
  render.def("mesh_confocal_grad_verts",
             &render::mesh_confocal_grad_verts);
  render.def("mesh_exhaustive_grad_verts",
             &render::mesh_exhaustive_grad_verts);
  render.def("mesh_exhaustive_forward",
             &render::mesh_exhaustive_forward);
  render.def("volume_confocal_forward",
             &render::volume_confocal_forward);
  render.def("volume_confocal_grad_volume",
             &render::volume_confocal_grad_volume);
  render.def("volume_confocal_grad_measurement_points",
             &render::volume_confocal_grad_measurement_points);
  render.def("volume_render_exhaustive_forward",
             &render::volume_render_exhaustive_forward);
  render.def("volume_render_exhaustive_grad_volume",
             &render::volume_render_exhaustive_grad_volume);

  // util
  py::module util = m.def_submodule("util");
  util.def("rbf2volume_gaussian_forward",
           &util::rbf2volume_gaussian_forward);
  util.def("rbf2volume_gaussian_grad",
           &util::rbf2volume_gaussian_grad);
}

} // namespace totri
