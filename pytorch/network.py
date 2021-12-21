"""Definition of the network architecture."""

import torch.nn as nn


class CNN(nn.Module):
    """A simple model

    Features 2 Conv layers, a feature-wise MaxPool and a Linear layer with
    Softmax activation.

    """

    def __init__(self):
        """Set up the layers of the model."""
        super().__init__()

        self.conv_relu_pool_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_relu_pool_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            # global max pooling, here: max pooling with a 7x7 kernel
            nn.MaxPool2d(7)
        )
        self.out = nn.LazyLinear(out_features=10)

    def forward(self, x):
        """The forward pass through the network."""
        x = self.conv_relu_pool_1(x)
        x = self.conv_relu_pool_2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)
