import torch
from torch import nn

class SE(nn.Module):
    def __init__(self, feature: int):
        super().__init__()
        self.gp = nn.AdaptiveAvgPool2d(1)
        #self.shape = (feature, 1, 1)
        self.dense = nn.Sequential(
            nn.Linear(in_features=feature, out_features=feature // 8, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=feature // 8, out_features=feature, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init = x

        x = self.gp(x)

        x = self.dense(x.flatten(start_dim=1))

        x = x.unsqueeze(-1).unsqueeze(-1)

        x = torch.mul(input=init,other=x)

        return x


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)
