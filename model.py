import torch
import torch.nn as nn
from monai.networks.nets import UNet as MONAI_UNet

class MedSegUNet(nn.Module):
    def __init__(self, pretrained_path=None):
        super(MedSegUNet, self).__init__()
        self.unet = MONAI_UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=32,
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            num_res_units=2,
            norm="batch",
            act="relu"
        )
        self.adapter = nn.Conv2d(32, 1, kernel_size=1, bias=True)

        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.unet.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded pretrained MedSeg UNet weights from {pretrained_path}")
        else:
            print("⚠️ No pretrained weights provided. Initializing UNet from scratch.")

        nn.init.kaiming_normal_(self.adapter.weight, mode='fan_out', nonlinearity='relu')
        if self.adapter.bias is not None:
            nn.init.zeros_(self.adapter.bias)

    def forward(self, x):
        x = self.unet(x)
        x = self.adapter(x)
        return x

def get_model(model_name="unet", pretrained=True, pretrained_path=None):
    if model_name.lower() == "unet":
        return MedSegUNet(pretrained_path=pretrained_path if pretrained else None)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Only 'unet' is supported.")