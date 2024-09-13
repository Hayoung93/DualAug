FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
USER root
RUN apt update && apt install git tmux nano -y
RUN conda create -n dg_clip python=3.8.18
RUN echo "source activate dg_clip" > ~/.bashrc
ENV PATH /opt/conda/envs/dg_clip/bin:$PATH
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
