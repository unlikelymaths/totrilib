FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel AS base
# Misc
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
      wget \
      libssl-dev \
      && \
    rm -rf /var/lib/apt/lists/*
# CMake
ARG CMAKE_BUILD_THREADS=24
WORKDIR /tmp/
RUN wget -P ./ https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1.tar.gz
RUN tar -xzf cmake-3.21.1.tar.gz
WORKDIR /tmp/cmake-3.21.1
RUN ./bootstrap --parallel=$CMAKE_BUILD_THREADS -- -DCMAKE_BUILD_TYPE:STRING=Release
RUN make -j $CMAKE_BUILD_THREADS install
RUN rm -rf /tmp/cmake-3.21.1
# Workdir
WORKDIR /workspace/totrilib

FROM base AS dev
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
      git \
      && \
    rm -rf /var/lib/apt/lists/*
