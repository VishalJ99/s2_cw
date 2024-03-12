FROM continuumio/miniconda3

RUN mkdir -p s2_cw

COPY . /s2_cw
WORKDIR /s2_cw

RUN conda env update --file environment.yml

RUN echo "conda activate s2_cw" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pre-commit install
