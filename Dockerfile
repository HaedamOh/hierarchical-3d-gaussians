ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
ARG USER_ID=1007
ARG GROUP_ID=1007
ENV DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME="/usr/local/cuda"

RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ca-certificates openssh-server vim ffmpeg libsm6 libxext6 python3-opencv gcc-11 g++-11 cmake

# Setup hierarchical_3d_gaussians
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0+PTX"

# Install COLMAP dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libmetis-dev \
    libflann-dev \
    libsqlite3-dev \
    libceres-dev \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    vim \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install GLOG (required by ceres).
RUN git clone --branch v0.6.0 https://github.com/google/glog.git --single-branch && \
    cd glog && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j "$(nproc)" && \
    make install && \
    cd ../.. && \
    rm -rf glog

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# Install Ceres-solver (required by colmap).
RUN git clone --branch 2.1.0 https://ceres-solver.googlesource.com/ceres-solver.git --single-branch && \
    cd ceres-solver && \
    git checkout "$(git describe --tags)" && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j "$(nproc)" && \
    make install && \
    cd ../.. && \
    rm -rf ceres-solver

# Install colmap.
RUN git clone --branch 3.8 https://github.com/colmap/colmap.git --single-branch && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. -DCUDA_ENABLED=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    make -j "$(nproc)" && \
    make install && \
    cd ../.. && \
    rm -rf colmap

ARG UID=1007
ARG GID=1007
ARG USERNAME=developer
RUN apt-get update \
&& apt-get install -y \
    sudo \
&& rm -rf /var/lib/apt/lists/* \
&& groupadd --gid 998 docker \
&& groupadd --gid 1013 oxford_spires \
&& groupadd --gid 1014 nerfstudio \
&& addgroup --gid ${GID} ${USERNAME} \
&& adduser --disabled-password --gecos '' --uid ${UID} --gid ${GID} ${USERNAME} \
&& usermod -aG docker,oxford_spires,nerfstudio ${USERNAME} \
&& echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} 
# && chown -R ${UID}:${GID} /home/${USERNAME} \
# && chown -R ${UID}:${GID} ${SOURCE_DIR}
    

WORKDIR /home/docker_dev
SHELL ["/bin/bash", "-c"]

RUN python3 -m pip install --upgrade pip setuptools pathtools promise pybind11

RUN CUDA_VER=${CUDA_VERSION%.*} && CUDA_VER=${CUDA_VER//./} && python3 -m pip install \
    torch==2.3.0+cu${CUDA_VER} \
    torchvision==0.18.0+cu${CUDA_VER} \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}


ARG GAUSSIAN_SPLATTING_DIR=/home/hierarchical-3d-gaussians
COPY . ${GAUSSIAN_SPLATTING_DIR}

WORKDIR ${GAUSSIAN_SPLATTING_DIR}
COPY ./submodules ${GAUSSIAN_SPLATTING_DIR}/submodules
# Install submodules. Permission issue or CUDA ENV, architecture issue 
# Set working directory for building the submodule
WORKDIR ${GAUSSIAN_SPLATTING_DIR}/submodules/gaussianhierarchy

# Build the submodule using CMake
RUN cmake . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j --config Release

# Switch back to the main directory
WORKDIR ${GAUSSIAN_SPLATTING_DIR}

# Install Python packages, including submodules
RUN pip install --no-cache-dir submodules/hierarchy-rasterizer && \
    pip install --no-cache-dir submodules/simple-knn && \
    pip install --no-cache-dir submodules/DPT 

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

WORKDIR /home/docker_dev


USER ${USERNAME}
RUN echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;33m\]\u@docker-\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> ~/.bashrc