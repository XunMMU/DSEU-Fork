import torch
from torch import nn

from .se import SE

class Decode(nn.Module):
    def __init__(self, in_ft:int, ot_ft:int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=in_ft, out_channels=ot_ft, kernel_size=2, padding="same"), # violation
            nn.BatchNorm2d(num_features=ot_ft),
            nn.ReLU()
        )
        self.se = SE(ot_ft)
        self.dec = nn.Sequential(
            nn.Conv2d(in_channels=in_ft, out_channels=ot_ft, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=ot_ft),
            nn.ReLU(),
            nn.Conv2d(in_channels=ot_ft, out_channels=ot_ft, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=ot_ft),
            nn.ReLU()
        )


    def forward(self, x: torch.Tensor, conv: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        conv = self.se(conv)

        x = torch.cat((conv,x), dim=1)
        x = self.dec(x)
        return x


    def __call__(self, x: torch.Tensor, conv: torch.Tensor) -> torch.Tensor:
        return super().__call__(x, conv)
