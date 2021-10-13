#!/bin/sh

# コンテナイメージ
IMAGE_NAME="cnn-gnn-classifier"

docker run --rm -it -v $(pwd):/workspace \
    -e http_proxy=http://proxy.noc.kochi-tech.ac.jp:3128 \
    -e https_proxy=http://proxy.noc.kochi-tech.ac.jp:3128 \
    --gpus all -p 8888:8888 ${IMAGE_NAME} \
    jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 \
    --allow-root --NotebookApp.token=""

	
