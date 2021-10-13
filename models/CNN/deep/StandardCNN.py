import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes, input_size):
        self.num_classes = num_classes
        self.input_size = input_size
        self.line_size = self.cal_middle_size(input_size)

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, kernel_size=5, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, kernel_size=3),
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

        out = _cal_covsize(input, 5, 4, 1)
        out = _cal_covsize(out, 5, 4, 1)
        out = _cal_covsize(out, 3, 0, 1)

        return int(out)
