import torch.nn as nn


class Simple_classifier(nn.Module):
    """単層の全結合分類層"""

    def __init__(self, inchannel, num_classes):
        super(Simple_classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(inchannel, num_classes), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class Complex_classifier(nn.Module):
    """3層の全結合分類層"""

    def __init__(self, inchannel, num_classes):
        super(Complex_classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(inchannel, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.classifier(x)
