# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ENV DEBCONF_NOWARNINGS yes
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
	wget since apt-utils

RUN apt-get install -y git libgl1-mesa-dev
RUN conda install -c conda-forge ipywidgets nodejs jupyter-lsp
RUN conda install -c conda-forge jupyterlab=2.1 ujson=1.35 jedi=0.15.2 parso=0.5.2
RUN pip install --upgrade pip
RUN pip install matplotlib networkx pandas seaborn sklearn opencv-python opencv-contrib-python
RUN pip install -i https://test.pypi.org/simple/ reserch-utils-HT

# dawnload torch geometric
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-geometric
