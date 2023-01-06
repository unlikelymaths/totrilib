#!/bin/bash
conda install -y \
  -c conda-forge \
  -c pytorch3d \
  ipympl jupyter matplotlib pytorch3d "tensorboard>=2.4.1"
pip install -e /workspace/totrilib/extern/tomcubes
pip install -e /workspace/totrilib
pip install opencv-python
apt-get update
apt-get install libgl-dev