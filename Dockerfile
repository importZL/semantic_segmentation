FROM pytorch/pytorch:latest
ENV PATH="/root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt install -y \
    htop \
    gcc \
    nano \
    curl \
    python3-pip \
    build-essential \
    git \
    curl \
    vim \
    tmux \
    wget \
    bzip2 \
    unzip \
    g++ \
    ca-certificates \
    ffmpeg \
    libx264-dev \
    imagemagick

RUN /bin/bash -c  "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh"

RUN conda create -n myenv python=3.8
ADD requirements.txt /root/requirements.txt
WORKDIR /root

RUN /bin/bash -c "source activate myenv \
    && pip install -r requirements.txt" 
CMD bash
