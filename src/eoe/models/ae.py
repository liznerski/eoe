import torch
import torch.nn as nn
import torch.nn.functional as F
from eoe.models.cnn import CNN32


class AE32(nn.Module):
    """ some autoencoder architecture for 32x32 images """

    def __init__(self, bias=True):
        super().__init__()
        self.bias = bias
        self.encoder = CNN32(bias)

        self.bn1d = nn.BatchNorm1d(128, eps=1e-04, affine=self.bias)
        self.deconv1 = nn.ConvTranspose2d(int(128 / (4 * 4)), 128, 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=self.bias)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=self.bias)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=self.bias)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=self.bias, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.encoder(x)
        x = self.bn1d(x)
        x = x.view(x.size(0), int(128 / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x
