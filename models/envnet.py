import torch
import torch.nn as nn
from modules import *

__all__ = ["create_model", "EnvNet3D"]

def create_model(cfg='config.yaml'):
    import yaml, os
    if isinstance(cfg, str) and os.path.exists(cfg):
        with open(cfg, 'r') as f:
            cfg = yaml.safe_load(f)
    return EnvNet3D(hidden_width=cfg['hidden_width'], padding_mode=cfg['padding_mode'], factorization=cfg['factorization'])


class EnvNet3D(nn.Module):
    def __init__(self, hidden_width=5, padding_mode='circular', factorization=False):
        super().__init__()
        # Backbone
        self.multiconv1 = MultiConv3d(2, hidden_width, 3, 7, 9, padding_mode, True, factorization=factorization)
        self.conv1 = ConvBN3d(hidden_width, hidden_width*2, 3, 2, padding_mode, factorization=factorization)
        self.multiconv2 = MultiConv3d(hidden_width*2, hidden_width*2, 3, 5, 7, padding_mode, True, factorization=factorization)
        self.conv2 = ConvBN3d(hidden_width*2, hidden_width*4, 3, 2, padding_mode, factorization=factorization)
        self.multiconv3 = MultiConv3d(hidden_width*4, hidden_width*4, 3, 3, 5, padding_mode, True, factorization=factorization)
        self.conv3 = ConvBN3d(hidden_width*4, hidden_width*2, 3, 1, padding_mode, factorization=factorization)

        # Neck
        self.upcatconv1 = UpCatConv3d(hidden_width*2, hidden_width, 3, 1, padding_mode, factorization=factorization)
        self.upcatconv2 = UpCatConv3d(hidden_width, hidden_width, 3, 1, padding_mode, factorization=factorization)
        self.downcatconv1 = DownCatConv3d(hidden_width, hidden_width*2, 3, 1, padding_mode, factorization=factorization)
        self.downcatconv2 = DownCatConv3d(hidden_width*2, hidden_width*4, 3, 1, padding_mode, factorization=factorization)

        # Head
        self.head1 = HeadConv3d(hidden_width, 4, 3, 1, padding_mode, factorization=factorization)
        self.head2 = HeadConv3d(hidden_width*2, 4, 3, 1, padding_mode, factorization=factorization)
        self.head3 = HeadConv3d(hidden_width*4, 4, 3, 1, padding_mode, factorization=factorization)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x1 = self.multiconv1(x)
        x2 = self.multiconv2(self.conv1(x1))
        x3 = self.multiconv3(self.conv2(x2))

        x3_2 = self.conv3(x3)
        x2_2 = self.upcatconv1(x3_2, x2)
        x1_2 = self.upcatconv2(x2_2, x1)

        x1_3 = x1_2
        x2_3 = self.downcatconv1(x1_3, x2_2)
        x3_3 = self.downcatconv2(x2_3, x3_2)

        out1 = self.head1(x1_3)
        out2 = self.head2(x2_3)
        out3 = self.head3(x3_3)

        return [out1, out2, out3]

if __name__ == "__main__":
    model = EnvNet3D(hidden_width=5)
    print(model)
    x = torch.randn(1, 2, 64, 64, 64)  # B, C, D, H, W
    y = model(x)
    print([o.shape for o in y])

    from thop import profile

    input = torch.randn(1, 2, 64, 64, 64)
    flops, params = profile(model, inputs=(input, ))
    print(f"FLOPs: {flops / 1e9:.2f} G, Params: {params / 1e6:.2f} M")