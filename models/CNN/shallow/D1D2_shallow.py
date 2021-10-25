import torch
import torch.nn as nn

from models.CNN.BrainNetCNN import E2EBlock
from models.CNN.D1D2Conv import D1D2Conv
from models.CNN.classifier import Simple_classifier, Complex_classifier


def shape_classifier(size, kernel, stride, padding=0):
    return int((((size + 2 * padding - (kernel - 1)) - 1) / stride) + 1)


class D1D2_model(nn.Module):
    def __init__(self, inception, classifier, num_classes, input_size):
        super().__init__()

        self.inception = inception

        inchannel = (
            self.inception(torch.ones(1, 1, input_size, input_size)).flatten().shape[0]
        )

        if classifier == "simple":
            self.classifier = Simple_classifier(inchannel, num_classes)
        elif classifier == "complex":
            self.classifier = Complex_classifier(inchannel, num_classes)

    def forward(self, x):
        x = self.inception(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
