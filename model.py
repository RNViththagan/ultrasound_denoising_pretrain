import torch
import torch.nn as nn
import torchvision.models as models
from monai.networks.nets import UNet as MONAI_UNet

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

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
            print("⚠️ No pretrained weights provided. Initializing MedSegUNet from scratch.")

        nn.init.kaiming_normal_(self.adapter.weight, mode='fan_out', nonlinearity='relu')
        if self.adapter.bias is not None:
            nn.init.zeros_(self.adapter.bias)

    def forward(self, x):
        x = self.unet(x)
        x = self.adapter(x)
        return x

class ModifiedResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ModifiedResNet, self).__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        new_weights = resnet.conv1.weight.mean(dim=1, keepdim=True)  # [64, 3, 7, 7] -> [64, 1, 7, 7]
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(new_weights)
       # ResNet encoder components
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # [batch_size, 256, 64, 64]
        self.layer2 = resnet.layer2  # [batch_size, 512, 32, 32]
        self.layer3 = resnet.layer3  # [batch_size, 1024, 16, 16]
        self.layer4 = resnet.layer4  # [batch_size, 2048, 8, 8]

        # Decoder for reconstruction (inspired by UNet)
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)  # [batch_size, 1024, 16, 16]
        self.dec4 = DoubleConv(2048, 1024)  # Concat with layer3: [2048, 16, 16] -> [1024, 16, 16]
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # [batch_size, 512, 32, 32]
        self.dec3 = DoubleConv(1024, 512)   # Concat with layer2: [1024, 32, 32] -> [512, 32, 32]
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # [batch_size, 256, 64, 64]
        self.dec2 = DoubleConv(512, 256)    # Concat with layer1: [512, 64, 64] -> [256, 64, 64]
        self.upconv1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)    # [batch_size, 64, 128, 128]
        self.dec1 = DoubleConv(64, 64)      # [batch_size, 64, 128, 128]
        self.final_upconv = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2) # [batch_size, 64, 256, 256]
        self.out = nn.Conv2d(64, 1, kernel_size=1)  # [batch_size, 1, 256, 256]

        if pretrained:
            print("✅ Loaded ImageNet pretrained weights for ResNet50 encoder.")
        else:
            print("⚠️ Initializing ModifiedResNet without pretrained weights.")

    def forward(self, x):
        # Encoder (ResNet)
        x = self.conv1(x)    # [batch_size, 64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [batch_size, 64, 64, 64]
        e1 = self.layer1(x)  # [batch_size, 256, 64, 64]
        e2 = self.layer2(e1) # [batch_size, 512, 32, 32]
        e3 = self.layer3(e2) # [batch_size, 1024, 16, 16]
        e4 = self.layer4(e3) # [batch_size, 2048, 8, 8]

        # Decoder
        d4 = self.upconv4(e4)              # [batch_size, 1024, 16, 16]
        d4 = torch.cat([d4, e3], dim=1)    # [batch_size, 1024+1024, 16, 16]
        d4 = self.dec4(d4)                 # [batch_size, 1024, 16, 16]
        d3 = self.upconv3(d4)              # [batch_size, 512, 32, 32]
        d3 = torch.cat([d3, e2], dim=1)    # [batch_size, 512+512, 32, 32]
        d3 = self.dec3(d3)                 # [batch_size, 512, 32, 32]
        d2 = self.upconv2(d3)              # [batch_size, 256, 64, 64]
        d2 = torch.cat([d2, e1], dim=1)    # [batch_size, 256+256, 64, 64]
        d2 = self.dec2(d2)                 # [batch_size, 256, 64, 64]
        d1 = self.upconv1(d2)              # [batch_size, 64, 128, 128]
        d1 = self.dec1(d1)                 # [batch_size, 64, 128, 128]
        d1 = self.final_upconv(d1)         # [batch_size, 64, 256, 256]
        return self.out(d1)

def get_model(model_name="unet", pretrained=True, pretrained_path=None):
    if model_name.lower() == "unet":
        return MedSegUNet(pretrained_path=pretrained_path)
    elif model_name.lower() == "resnet":
        return ModifiedResNet(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'unet' or 'resnet'.")