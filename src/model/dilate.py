import torch
from torch import nn

class Dilate(nn.Module):
    def __init__(self, d_num: int):
        super().__init__()
        self.dil  = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, dilation=d_num, padding="same"),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dil(x)
        return x


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)
