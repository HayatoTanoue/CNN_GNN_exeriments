import torch
import torch.nn as nn

from models.CNN.BrainNetCNN import E2EBlock
from models.CNN.classifier import Simple_classifier, Complex_classifier


class D1D2Conv(nn.Module):
    """カーネルサイズ3,5,7 の2D conv + E2E conv"""

    def __init__(self, in_channel, out_channel, resize, pool, use_attention=True):
        super(D1D2Conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.ReLU(),
        )

        self.e2econv = nn.Sequential(
            E2EBlock(
                in_channel, out_channel, torch.ones(1, 1, resize, resize), bias=True
            ),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(pool[0], pool[1])

        self.use_attention = use_attention
        if use_attention:
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
        if self.use_attention:
            outputs = outputs * self.softmax(self.attention)

        return outputs


def shape_classifier(size, kernel, stride, padding=0):
    return int((((size + 2 * padding - (kernel - 1)) - 1) / stride) + 1)


class Deep_D1D2_feature(nn.Module):
    """
    3層のD1D2 conv
    """

    def __init__(self, resize, use_attention=True):
        super(Deep_D1D2_feature, self).__init__()
        self.calc_input_size1 = shape_classifier(resize, 2, 2)
        self.calc_input_size2 = shape_classifier(self.calc_input_size1, 2, 2)
        self.d1d2_conv1 = D1D2Conv(1, 16, resize, (2, 2), use_attention)
        self.d1d2_conv2 = D1D2Conv(64, 8, self.calc_input_size1, (2, 2), use_attention)
        self.d1d2_conv3 = D1D2Conv(32, 4, self.calc_input_size2, (2, 2), use_attention)

    def forward(self, x):
        x = self.d1d2_conv1(x)
        x = self.d1d2_conv2(x)
        x = self.d1d2_conv3(x)
        return x


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
