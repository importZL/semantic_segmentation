FROM pytorch/pytorch:latest
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt install -y \
    htop wget gcc git nano curl

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     # && mkdir /root/miniconda3 \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b 
# && rm -f Miniconda3-latest-Linux-x86_64.sh

# RUN conda create -y -n myenv python=3.8

RUN /bin/bash -c "pip install wandb imgaug pandas scikit-learn wandb transformers diffusers accelerate"

CMD bash
