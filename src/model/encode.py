from typing import Tuple
import torch
from torch import nn

class Encode(nn.Module):
    def __init__(self, in_ft:int, ot_ft:int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=in_ft, out_channels=ot_ft, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=ot_ft),
            nn.ReLU(),
            nn.Conv2d(in_channels=ot_ft, out_channels=ot_ft, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=ot_ft),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        con = self.enc(x)
        x = self.pool(con)
        return x, con


    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(x)
