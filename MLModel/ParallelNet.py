# -*- coding: utf-8 -*-
# Convolution neural network
import torch
from torch import nn
import torch.nn.functional as F 

class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # Concatenate the outputs on the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)
        
class ParallelNet(nn.Module):
    def __init__(self, Inception=Inception, output_size=6):
        super(ParallelNet, self).__init__()
        self.b0 = nn.Sequential(nn.BatchNorm2d(1))
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 192, kernel_size=3, padding=1),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                        Inception(256, 128, (128, 192), (32, 96), 64),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                        Inception(512, 160, (112, 224), (24, 64), 64),
                        Inception(512, 128, (128, 256), (24, 64), 64),
                        Inception(512, 112, (144, 288), (32, 64), 64),
                        Inception(528, 256, (160, 320), (32, 128), 128),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                        Inception(832, 384, (192, 384), (48, 128), 128),
                        nn.AdaptiveMaxPool2d((1,1)),
                        nn.Flatten())
        self.dense = nn.Sequential(nn.Linear(1024, output_size),
                                    nn.Sigmoid())

    def forward(self, x):
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        y = self.dense(x)
        return(y)

if __name__ == "__main__":
    model = ParallelNet(Inception=Inception, output_size=5)
    X = torch.rand(size=(1, 1, 3, 20))
    Y = model(X)
    print(Y.size())
    print(Y)


