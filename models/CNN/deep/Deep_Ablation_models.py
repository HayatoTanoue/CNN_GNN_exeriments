import torch
import torch.nn as nn

from models.CNN.BrainNetCNN import E2EBlock
from models.CNN.D1D2Conv import D1D2Conv
from models.CNN.classifier import Simple_classifier, Complex_classifier


class Select_KernelSize_CNN(nn.Module):
    """3層, 同一カーネルサイズ"""

    def __init__(self, kernel_size, num_classes, input_size):
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.padding = kernel_size // 2

        self.line_size = self.cal_middle_size(input_size)

        super(Select_KernelSize_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, self.kernel_size, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, self.kernel_size, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, self.kernel_size, stride=1, padding=self.padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * self.line_size * self.line_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def cal_middle_size(self, input):
        """3回畳み込みとプーリングを行ったときのサイズを計算"""

        def _cal_covsize(height, kernel_size, padding, stride):
            out = (height + 2 * padding - (kernel_size - 1)) / stride
            out /= 2
            return out

        out = _cal_covsize(input, self.kernel_size, self.padding, 1)
        out = _cal_covsize(out, self.kernel_size, self.padding, 1)
        out = _cal_covsize(out, self.kernel_size, self.padding, 1)

        return int(out)


class Only_E2E_model(nn.Module):
    def __init__(self, num_classes, input_size):
        super(Only_E2E_model, self).__init__()

        self.num_classes = num_classes
        self.second_input_size = input_size // 2
        self.third_input_size = self.second_input_size // 2
        self.out_put_size = self.third_input_size // 2

        self.features = nn.Sequential(
            E2EBlock(1, 64, torch.ones(1, 1, input_size, input_size), bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            E2EBlock(
                64,
                32,
                torch.ones(1, 1, self.second_input_size, self.second_input_size),
                bias=True,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            E2EBlock(
                32,
                16,
                torch.ones(1, 1, self.third_input_size, self.third_input_size),
                bias=True,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * self.out_put_size * self.out_put_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, self.num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def shape_classifier(size, kernel, stride, padding=0):
    return int((((size + 2 * padding - (kernel - 1)) - 1) / stride) + 1)


class D1D2_shallow_layer(nn.Module):
    """
    層の数可変D1D2
    """

    def __init__(
        self, resize, num_classes, num_layer, use_attention=True, classifier="complex"
    ):
        super(D1D2_shallow_layer, self).__init__()
        layers = []
        # 1層目の追加
        layers.append(D1D2Conv(1, 8, resize, (2, 2), use_attention))
        calc_input_size1 = shape_classifier(resize, 2, 2)
        # 2層目以降の追加
        for i in range(num_layer):
            layers.append(D1D2Conv(32, 8, calc_input_size1, (2, 2), use_attention))
            calc_input_size1 = shape_classifier(calc_input_size1, 2, 2)

        self.inception = nn.Sequential(*layers)

        if classifier == "simple":
            self.classifier = Simple_classifier(
                32 * calc_input_size1 * calc_input_size1, num_classes
            )
        elif classifier == "complex":
            self.classifier = Complex_classifier(
                32 * calc_input_size1 * calc_input_size1, num_classes
            )

    def forward(self, x):
        x = self.inception(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
