FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04


ENV DEBIAN_FRONTEND=noninteractive

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8


RUN apt-get update && apt-get install -y \
    git xvfb \
    libglu1-mesa libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev mesa-utils freeglut3 freeglut3-dev \
    libglfw3 libglfw3-dev zlib1g zlib1g-dev libsdl2-dev libjpeg-dev lua5.1 liblua5.1-0-dev libffi-dev \
    build-essential cmake pkg-config software-properties-common gettext \
    ffmpeg patchelf swig unrar unzip zip curl wget tmux

# System packages.
RUN apt-get update && apt-get install -y \
  ffmpeg \
  mesa-utils \
  git \
  libgl1-mesa-dev \
  python3-pip \
  unrar \
  wget \
  && apt-get clean

# Python packages.
RUN pip3 install --no-cache-dir \
  opencv-contrib-python \
  pyvirtualdisplay \
  gym==0.21.0 \
  ruamel.yaml \
  tensorflow==2.6.0 \
  pybullet \
  protobuf==3.19.0 \
  numpy==1.23.5 \
  tensorflow_probability==0.12.2 \
  keras==2.6 \
  Image

# DreamerV2.
ENV TF_XLA_FLAGS --tf_xla_auto_jit=2
WORKDIR /app