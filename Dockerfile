FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y \
    ca-certificates \
    git \
    python3.10 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /dfanalyzer

COPY . .

RUN pip install --upgrade pip && \
    pip install build setuptools streamlit wheel && \
    pip install .

ENTRYPOINT ["dfanalyzer"]
