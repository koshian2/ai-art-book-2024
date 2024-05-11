# PyTorchのバージョンに合わせてCUDAのイメージを変える
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update
ENV TZ=Asia/Tokyo
ENV LANG=en_US.UTF-8
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -yq --no-install-recommends python3-pip \
        python3-dev \
        wget \
        git  \
        libopencv-dev \
        tzdata && apt-get upgrade -y && apt-get clean

RUN ln -s /usr/bin/python3 /usr/bin/python

# PyTorchのバージョンは適宜変えてね
RUN pip install -U pip &&\
  pip install --no-cache-dir torch==2.3.0+cu121 torchvision==0.18.0+cu121 xformers==0.0.26.post1 --extra-index-url https://download.pytorch.org/whl/cu121
RUN git clone https://github.com/kohya-ss/sd-scripts.git
WORKDIR sd-scripts
RUN pip install --no-cache-dir -r requirements.txt
# for WD 1.4 Tagger
RUN pip install --no-cache-dir onnx==1.16.0 onnxruntime-gpu==1.17.1 