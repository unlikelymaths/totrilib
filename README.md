# totrilib - Torch Transient Imaging Library

Official implementation of **"Fast Differentiable Transient Rendering for Non-Line-of-Sight Reconstruction"**
[[paper]](https://openaccess.thecvf.com/content/WACV2023/papers/Plack_Fast_Differentiable_Transient_Rendering_for_Non-Line-of-Sight_Reconstruction_WACV_2023_paper.pdf)

## Install

Totrilib in only tested on Linux,
but should also work on Windows.

### Docker

Make sure you have
[Docker](https://www.docker.com/)
and 
[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
installed.
Download NVIDIA Optix (7.2.0) from
<https://developer.nvidia.com/designworks/optix/downloads/legacy>
and extract to `extern/NVIDIA-OptiX-SDK-7.2.0`,
and make sure you have [pybind11](https://github.com/pybind/pybind11)
in `extern/pybind11`.
The easiest way to run totrilib is to use
[Docker Compose](https://docs.docker.com/compose/)
and start a detached container with
`docker compose up -d`.
Otherwise see `docker-compose.yml` for settings to start a container.
Mount additional volumes as needed.

### Standard

Make sure you have the following dependencies installed.
The suggested versions should work fine, but others might also do.

- CMake (3.21.1)
- NVIDIA CUDA (11.1) and NVIDIA driver (470.57.02)
- NVIDIA Optix (7.2.0)  
  Download from
  <https://developer.nvidia.com/designworks/optix/downloads/legacy>.
  Make sure to set environment variable `OptiX_INSTALL_DIR` to your install
  directory or install in `extern/NVIDIA-OptiX-SDK-7.2.0`.
- pybind11  
  Download from <https://github.com/pybind/pybind11> to `extern/pybind11`
- Python (3.7.10)
- Pytorch (1.9.0)

Install with pip (`pip install /path/to/totrilib`).

## Samples

Code will follow soon.

To run the sample code, various data files are required.
For the *Statue*, *Diffuse S*, *Mannequin* and the *Zaragoza Bunny*
we refer to the respective publications listed below.
Adjust the paths in the sample files as needed.

- Andreas Velten, Thomas Willwacher, Otkrist Gupta, Ashok Veeraraghavan,
  Moungi G Bawendi, Ramesh Raskar, 2012,
  "*Recovering three-dimensional shape around a corner using ultrafast
  time-of-flight imaging*",
  <https://www.nature.com/articles/ncomms1747>
- Matthew O’Toole, David B Lindell, Gordon Wetzstein, 2018,
  "*Confocal non-line-of-sight imaging based on the light-cone transform*",
  <https://www.computationalimaging.org/publications/confocal-non-line-of-sight-imaging-based-on-the-light-cone-transform/>
- David B Lindell, Gordon Wetzstein, Matthew O'Toole, 2019,
  "*Wave-based non-line-of-sight imaging using fast fk migration*",
  <http://www.computationalimaging.org/publications/nlos-fk/>
- Miguel Galindo, Julio Marco, Matthew O'Toole, Gordon Wetzstein,
  Diego Gutierrez, Adrian Jarabo, 2019,
  "*A dataset for benchmarking time-resolved non-line-of-sight imaging*",
  <https://graphics.unizar.es/nlos_dataset>

## Citation
If you find this repository helpful please cite the following paper.
```bibtex
  @inproceedings{plack2023fast,
    title={Fast Differentiable Transient Rendering for Non-Line-of-Sight Reconstruction},
    author={Plack, Markus and Callenberg, Clara and Schneider, Monika and Hullin, Matthias B},
    booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
    pages={3067--3076},
    year={2023}
  }
```