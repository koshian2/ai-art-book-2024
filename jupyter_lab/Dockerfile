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

COPY requirements.txt .
RUN pip install -U pip &&\
  pip install --no-cache-dir -r requirements.txt
WORKDIR /home
EXPOSE 8888
CMD ["jupyter", "lab", "--notebook-dir=/home", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''"]