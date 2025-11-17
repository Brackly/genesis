import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Degrades/encodes the image progressively"""

    def __init__(self, bottleneck_dim=256):
        super().__init__()
        # Progressive downsampling
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)  # 64->32
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 32->16
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 16->8
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)  # 8->4

        # Bottleneck
        self.bottleneck = nn.Conv2d(256, bottleneck_dim, 4)  # 4->1

    def forward(self, x):
        # Store intermediate features for skip connections if desired
        features = []
        x = torch.relu(self.conv1(x))
        features.append(x)
        x = torch.relu(self.conv2(x))
        features.append(x)
        x = torch.relu(self.conv3(x))
        features.append(x)
        x = torch.relu(self.conv4(x))
        features.append(x)

        bottleneck = self.bottleneck(x)
        return bottleneck, features


class Decoder(nn.Module):
    """Attempts to reconstruct from the degraded representation"""

    def __init__(self, bottleneck_dim=256):
        super().__init__()
        # Reverse the bottleneck
        self.unbottleneck = nn.ConvTranspose2d(bottleneck_dim, 256, 4)

        # Progressive upsampling (transpose convolutions)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)

    def forward(self, bottleneck):
        x = torch.relu(self.unbottleneck(bottleneck))
        x = torch.relu(self.deconv4(x))
        x = torch.relu(self.deconv3(x))
        x = torch.relu(self.deconv2(x))
        x = torch.tanh(self.deconv1(x))  # or sigmoid for [0,1] range
        return x

class VanillaVAE(nn.Module):
    """Vanilla VAE"""
    def __init__(self, bottleneck_dim=256):
        super().__init__()
        self.encoder = Encoder(bottleneck_dim)
        self.decoder = Decoder(bottleneck_dim)
    def forward(self, x):
        bottleneck, features = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        return reconstruction
