FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu18.04
MAINTAINER MichaelYin<zhuoyin94@163.com>

# 设置环境变量与工作目录s
ENV PATH /opt/conda/bin:$PATH
ARG DEBIAN_FRONTEND=noninteractive

ADD . /
WORKDIR /

COPY /cached_pkgs/anaconda3-202105.sh /home
COPY /cached_pkgs/sources.list /etc/apt/sources.list

# apt-get升级 && 安装基础依赖
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends zip build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 构建anaconda基础环境
RUN set -x && \
    mkdir -p /opt && \
    /bin/bash /home/anaconda3-202105.sh -b -p /opt/conda && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    rm /home/anaconda3-202105.sh

# 安装requirements.txt
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip install ./cached_pkgs/tensorflow-2.8.0-cp38-cp38-manylinux2010_x86_64.whl && \
    pip install ./cached_pkgs/xgboost-1.6.0-py3-none-manylinux2014_x86_64.whl && \
    pip --no-cache-dir install -r requirements.txt --timeout 600 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    rm -rf ./cached_pkgs


