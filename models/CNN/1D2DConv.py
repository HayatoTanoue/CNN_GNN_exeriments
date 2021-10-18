import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.activation import ReLU


class E2EBlock(torch.nn.Module):
    """E2Eblock."""

    def __init__(self, in_planes, planes, example, bias=False):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class Conv_and_e2e(nn.Module):
    def __init__(self, out_channel):
        super(Conv_and_e2e, self).__init__()
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

        self.e2econv = nn.Sequential(
            E2EBlock(1, out_channel, torch.ones(1, 1, 100, 100), bias=True),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(10, 10)

        self.attention = nn.Parameter(
            torch.randn(out_channel * 3, 1, 1), requires_grad=True
        )
        self.softmax = nn.Softmax(dim=0)

        self.out_channel = out_channel * 4

    def forward(self, x):
        branch1 = self.pool(self.conv1(x))
        branch2 = self.pool(self.conv2(x))
        branch3 = self.pool(self.e2econv(x))

        outputs = [branch1, branch2, branch3]
        outputs = torch.cat(outputs, 1)
        outputs = outputs * self.softmax(self.attention)

        return outputs


class Simple_classifier(nn.Module):
    def __init__(self, inchannel, num_classes):
        super(Simple_classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(inchannel, num_classes), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class Model(nn.Module):
    def __init__(self, inception, classifier):
        super().__init__()

        self.inception = inception
        self.classifier = classifier

    def forward(self, x):
        x = self.inception(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
