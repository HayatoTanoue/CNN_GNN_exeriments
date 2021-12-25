# CNN base
from models.CNN.deep.StandardCNN import CNN
from models.CNN.BrainNetCNN import BrainNetCNN
from models.CNN.deep.Deep_Ablation_models import Select_KernelSize_CNN, Only_E2E_model
from models.CNN.deep.Deep_Ablation_models import D1D2_shallow_layer


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


def Ablation_study(model_name, num_class, input_size):
    """for abaltion study model"""
    if model_name == "Deep3":
        model = Select_KernelSize_CNN(3, num_class, input_size)
    elif model_name == "Deep5":
        model = Select_KernelSize_CNN(5, num_class, input_size)
    elif model_name == "Deep7":
        model = Select_KernelSize_CNN(7, num_class, input_size)
    elif model_name == "Only_E2E":
        model = Only_E2E_model(num_class, input_size)

    return model


def Ablation_shallow_D1D2(model_name, num_class, input_size):
    if model_name == "layer1_attention_complex":
        model = D1D2_shallow_layer(input_size, num_class, 1, True, "complex")
    elif model_name == "layer1_attention_simple":
        model = D1D2_shallow_layer(input_size, num_class, 1, True, "simple")
    elif model_name == "layer2_attention_complex":
        model = D1D2_shallow_layer(input_size, num_class, 2, True, "complex")
    elif model_name == "layer2_attention_simple":
        model = D1D2_shallow_layer(input_size, num_class, 2, True, "simple")
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
    if model_name == "Deep_D1D2_noAttention":
        print("no attention")
        feature = Deep_D1D2_feature(input_size, False)
    model = D1D2_model(feature, "complex", num_class, input_size)
    return model


def model_parameter_counter(model):
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    return num_params
