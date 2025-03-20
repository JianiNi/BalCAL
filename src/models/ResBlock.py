import torch.nn as nn
import torch.nn.functional as F
class ResBlock(nn.Module):
    def __init__(self, in_size:int, hidden_size:int, out_size:int, stride:int):
        super().__init__()
        self.skip = nn.Sequential()

        if stride != 1 or in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_size))
        else:
            self.skip = None

        self.batchnorm1 = nn.BatchNorm2d(in_size)
        self.conv1 = nn.Conv2d(in_size, hidden_size, kernel_size=3, stride=stride, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, out_size, kernel_size=3, padding=1)


    def convblock(self, x):
        x = self.conv1(F.relu(self.batchnorm1(x)))
        x = self.conv2(F.relu(self.batchnorm2(x)))
        return x

    def forward(self, x):
        if self.skip is not None:
            return self.skip(x) + self.convblock(x) # skip connection
        else:
            return x + self.convblock(x)