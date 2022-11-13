# ==================================================================
# module list
# ------------------------------------------------------------------
# jupyter hub+lab           latest (pip)
# pytorch                   latest (pip)
# ax                        latest (pip)
# tensorflow +(keras-tuner) latest (pip)
# NLP (spacy, nltk)         latest (pip)
# opencv                    4.1.1  (git)
# OpenAI gym                latest (pip)
# MLflow                    latest (pip)
# DVC                       latest (pip)
# Dask                      latest (pip)
# Ray                       latest (pip)
# Prefect                   latest (pip)
# ==================================================================

FROM ubuntu:20.04


RUN apt update && apt install -y python3.8 && apt install -y python3-pip

ENV DEBIAN_FRONTEND noninteractive
# ==================================================================
# python
# ------------------------------------------------------------------
RUN python3 -m pip install \
        setuptools \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-learn \
        joblib \
        matplotlib \
        Cython \
        h5py \
        onnx onnxruntime \ 
        opencv-python

# ==================================================================
# jupyter hub
# ------------------------------------------------------------------
# RUN eval apt install -y \
#     npm  nodejs && \
#     npm install -g configurable-http-proxy && \
RUN eval python3 -m pip install \
        jupyterhub jupyterlab && \
    mkdir -p /etc/jupyterhub
COPY configs/jupyterhub_config.py /etc/jupyterhub/jupyterhub_config.py

# ==================================================================
# pytorch
# ------------------------------------------------------------------
# ENV TORCHVISION_VERSION=0.6.1
# ENV TORCH_VERSION=1.5.1
# RUN python3 -m pip install \
# 		torch==$TORCH_VERSION+cpu torchvision==$TORCHVISION_VERSION+cpu -f https://download.pytorch.org/whl/torch_stable.html

# ==================================================================
# ax
# ------------------------------------------------------------------
RUN python3 -m pip install \
        ax-platform

# ==================================================================
# tensorflow with keras tuner
# ------------------------------------------------------------------
RUN python3 -m pip install \
        tensorflow keras-tuner

# ==================================================================
# NLP tools
# ------------------------------------------------------------------
RUN python3 -m pip install \
        nltk spacy fuzzywuzzy[speedup] gensim && \
        python3 -m nltk.downloader popular -d /usr/share/nltk_data && \
        python3 -m spacy download en

# ==================================================================
# opencv
# ------------------------------------------------------------------
RUN eval apt install -y \
        libatlas-base-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler 
#     $GIT_CLONE --branch 4.4.0 https://github.com/opencv/opencv ~/opencv && \
#     mkdir -p ~/opencv/build && cd ~/opencv/build && \
#     cmake -D CMAKE_BUILD_TYPE=RELEASE \
#           -D CMAKE_INSTALL_PREFIX=/usr/local \
#           -D WITH_IPP=OFF \
#           -D WITH_CUDA=OFF \
#           -D WITH_OPENCL=OFF \
#           -D BUILD_TESTS=OFF \
#           -D BUILD_PERF_TESTS=OFF \
#           .. && \
#     make -j"$(nproc)" install && \
#     ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2

# ==================================================================
# OpenAI GYM
# ------------------------------------------------------------------
RUN eval apt install -y \
        python3-dev \
        zlib1g-dev \
        libjpeg-dev \
        cmake \
        swig \
        python3-pyglet \
        python3-opengl \
        libboost-all-dev \
        libsdl2-dev \
        libosmesa6-dev \
        patchelf \
        ffmpeg \
        xvfb \
        && \
    python3 -m pip install \
        gym \
		'gym[atari]' \
		'gym[box2d]'

# ==================================================================
# MLflow
# ------------------------------------------------------------------
# RUN python3 -m pip install \
# 		mlflow && \
# 		sed -i 's/127.0.0.1/0.0.0.0/g' /usr/local/lib/python${PYTHON_COMPAT_VERSION}/dist-packages/mlflow/cli.py && \
#         curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#         bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b && \
#         rm Miniconda3-latest-Linux-x86_64.sh
# ENV PATH=${PATH}:/miniconda/bin
# RUN conda init && \
#         conda config --set auto_activate_base false

# ==================================================================
# DVC
# ------------------------------------------------------------------
RUN python3 -m pip install --upgrade pip && \
        python3 -m pip install dvc mlflow

# ==================================================================
# Dask
# ------------------------------------------------------------------
RUN python3 -m pip install \
        "dask[complete]"

# ==================================================================
# Ray
# ------------------------------------------------------------------
RUN python3 -m pip install \
        ray ray[debug]

# ==================================================================
# Prefect
# ------------------------------------------------------------------
RUN python3 -m pip install \
        prefect

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
RUN ldconfig && \
    apt-get clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# add default user
# ENV DEFAULT_USER=dlenv
# RUN chmod +x add-user.sh && ./add-user.sh $DEFAULT_USER

# make root dir owned by that user
# RUN chown -R $DEFAULT_USER:$DEFAULT_USER $root_HOME

# make jupyter notebook token equal to username by default
# ENV JUPYTER_LAB_TOKEN=$DEFAULT_USER

# copy run scripts
COPY scripts/run-* /
RUN chmod +x /run-*

# run as non-root
# USER $DEFAULT_USER

# make sure data folder has proper permissions
RUN mkdir -p /home/workspace/scripts


WORKDIR /home/workspace/scripts

# VOLUME /home/workspace/scripts

# mlflow 
EXPOSE 5000
# jupyterlab
EXPOSE 8888
# jupyterhub
EXPOSE 8000
# spark ui
EXPOSE 4040
# spark master
EXPOSE 7077
# spark worker
EXPOSE 8081
