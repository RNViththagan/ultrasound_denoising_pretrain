import torch
import torch.nn as nn
import torchvision.models as models

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

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = DoubleConv(1, 64)    # Input: [batch_size, 1, 256, 256], Output: [batch_size, 64, 256, 256]
        self.enc2 = DoubleConv(64, 128)  # Input: [batch_size, 64, 128, 128], Output: [batch_size, 128, 128, 128]
        self.enc3 = DoubleConv(128, 256) # Input: [batch_size, 128, 64, 64], Output: [batch_size, 256, 64, 64]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)  # Input: [batch_size, 256, 32, 32], Output: [batch_size, 512, 32, 32]

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # Input: [batch_size, 512, 32, 32], Output: [batch_size, 256, 64, 64]
        self.dec3 = DoubleConv(512, 256)  # Input: [batch_size, 512, 64, 64] (after concat with enc3), Output: [batch_size, 256, 64, 64]
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # Input: [batch_size, 256, 64, 64], Output: [batch_size, 128, 128, 128]
        self.dec2 = DoubleConv(256, 128)  # Input: [batch_size, 256, 128, 128] (after concat with enc2), Output: [batch_size, 128, 128, 128]
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Input: [batch_size, 128, 128, 128], Output: [batch_size, 64, 256, 256]
        self.dec1 = DoubleConv(128, 64)   # Input: [batch_size, 128, 256, 256] (after concat with enc1), Output: [batch_size, 64, 256, 256]

        # Output layer
        self.out = nn.Conv2d(64, 1, kernel_size=1)  # Input: [batch_size, 64, 256, 256], Output: [batch_size, 1, 256, 256]

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # [batch_size, 64, 256, 256]
        enc2 = self.enc2(self.pool(enc1))  # Pool: [batch_size, 64, 128, 128] -> Enc2: [batch_size, 128, 128, 128]
        enc3 = self.enc3(self.pool(enc2))  # Pool: [batch_size, 128, 64, 64] -> Enc3: [batch_size, 256, 64, 64]

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))  # Pool: [batch_size, 256, 32, 32] -> Bottleneck: [batch_size, 512, 32, 32]

        # Decoder
        dec3 = self.up3(bottleneck)  # [batch_size, 256, 64, 64]
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))  # Concat: [batch_size, 256+256, 64, 64] -> [batch_size, 256, 64, 64]
        dec2 = self.up2(dec3)  # [batch_size, 128, 128, 128]
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))  # Concat: [batch_size, 128+128, 128, 128] -> [batch_size, 128, 128, 128]
        dec1 = self.up1(dec2)  # [batch_size, 64, 256, 256]
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))  # Concat: [batch_size, 64+64, 256, 256] -> [batch_size, 64, 256, 256]

        # Output
        return self.out(dec1)  # [batch_size, 1, 256, 256]

class ModifiedResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ModifiedResNet, self).__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Modify first conv layer for 1-channel input
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            # Average the 3-channel weights across channels for 1-channel input
            original_weights = resnet.conv1.weight
            new_weights = original_weights.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
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
        out = self.out(d1)                 # [batch_size, 1, 256, 256]
        return out

def get_model(model_name="resnet", pretrained=True):
    """
    Returns the specified model.
    Args:
        model_name (str): 'resnet' for ModifiedResNet, 'unet' for UNet.
        pretrained (bool): Whether to load pretrained weights.
    Returns:
        nn.Module: The selected model.
    """
    if model_name.lower() == "resnet":
        return ModifiedResNet(pretrained=pretrained)
    elif model_name.lower() == "unet":
        return UNet()
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'resnet' or 'unet'.")