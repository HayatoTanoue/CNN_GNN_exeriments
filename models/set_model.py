# CNN base
from models.CNN.deep.StandardCNN import CNN
from models.CNN.BrainNetCNN import BrainNetCNN

# GNN base
from models.GNN.DeepMultisets import DeepMultisets
from models.GNN.DGCNN import DGCNN_Model
from models.GNN.GIN import GIN

# 1D2D base
from models.CNN.D1D2Conv import Deep_D1D2_feature, D1D2_model

import torch


def CNN_base(model_name, num_class, input_size):
    """set Deep CNN base model"""
    if model_name == "CNN":
        model = CNN(num_class, input_size)
    elif model_name == "BrainCNN":
        sample = torch.ones(1, 1, input_size, input_size)
        model = BrainNetCNN(sample, num_class)

    return model


def GNN_base(model_name, num_class, num_node_feature):
    """set GNN base model"""
    if model_name == "DGCNN":
        model = DGCNN_Model(num_node_feature, num_class)
    elif model_name == "GIN":
        model = GIN(
            num_node_feature,
            num_class,
            config={
                "aggregation": "sum",
                "dropout": 0.5,
                "hidden_units": [64, 64, 64, 64],
                "train_eps": True,
            },
        )
    elif model_name == "Deepsets":
        model = DeepMultisets(num_node_feature, num_class)
    return model


def D1D2_base(model_name, input_size, num_class, use_attention):
    if model_name == "Deep_D1D2":
        feature = Deep_D1D2_feature(input_size, use_attention)
        model = D1D2_model(feature, "complex", num_class, input_size)
    return model


def model_parameter_counter(model):
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    return num_params
