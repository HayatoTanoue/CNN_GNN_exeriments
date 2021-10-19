import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.activation import ReLU

from models.CNN.BrainNetCNN import E2EBlock
from models.CNN.classifier import Simple_classifier, Complex_classifier


class D1D2Conv(nn.Module):
    """カーネルサイズ3,5,7 の2D conv + E2E conv"""

    def __init__(self, out_channel):
        super(D1D2Conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channel,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channel,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.ReLU(),
        )

        self.e2econv = nn.Sequential(
            E2EBlock(1, out_channel, torch.ones(1, 1, 100, 100), bias=True),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(10, 10)

        self.attention = nn.Parameter(
            torch.randn(out_channel * 4, 1, 1), requires_grad=True
        )
        self.softmax = nn.Softmax(dim=0)

        self.out_channel = out_channel * 4

    def forward(self, x):
        branch1 = self.pool(self.conv1(x))
        branch2 = self.pool(self.conv2(x))
        branch3 = self.pool(self.conv3(x))
        branch4 = self.pool(self.e2econv(x))

        outputs = [branch1, branch2, branch3, branch4]
        outputs = torch.cat(outputs, 1)
        outputs = outputs * self.softmax(self.attention)

        return outputs


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
