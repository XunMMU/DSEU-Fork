from torchinfo import summary
import torch
import torch.nn as nn

from .encode import Encode
from .dilate import Dilate
from .decode import Decode

class DSEU(nn.Module):

    def __init__(self):
        super().__init__()

        # Encode
        self.enc1 = Encode(in_ft=3,ot_ft=64)
        self.enc2 = Encode(in_ft=64,ot_ft=128)
        self.enc3 = Encode(in_ft=128,ot_ft=256)
        self.enc4 = Encode(in_ft=256,ot_ft=512)
        # Neck
        self.d1f = Dilate(1)
        self.d2f = Dilate(2)
        self.d4f = Dilate(4)
        self.d8f = Dilate(8)
        # Decode
        self.dec1 = Decode(in_ft=1024,ot_ft=512)
        self.dec2 = Decode(in_ft=512,ot_ft=256)
        self.dec3 = Decode(in_ft=256,ot_ft=128)
        self.dec4 = Decode(in_ft=128,ot_ft=64)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1),
            nn.Tanh()
        )


    def forward(self, x:torch.Tensor): # (batch, 3, 256, 256)

        x, cov1 = self.enc1(x)
        x, cov2 = self.enc2(x)
        x, cov3 = self.enc3(x)
        x, cov4 = self.enc4(x)

        d1 = self.d1f(x)
        d2 = self.d2f(x)
        d3 = self.d4f(x)
        d4 = self.d8f(x)

        x = torch.cat((d1,d2,d3,d4), dim=1)

        x = self.dec1(x,cov4)
        x = self.dec2(x,cov3)
        x = self.dec3(x,cov2)
        x = self.dec4(x,cov1)

        x = self.final(x)

        return x


if __name__ == "__main__":
    summary(model=DSEU(), input_size=(6,3,256,256)) # test schema
