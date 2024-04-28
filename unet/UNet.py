import torch
import torch.nn as nn
import torchvision.transforms.functional as f

class conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x
    

class upConv(nn.Module):
    def __init__(self, scale_factor, mode, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor, mode)
        )

    def forward(self, x):
        x = self.upconv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = conv2D(in_channels, 64, 3, 1, 0)
        self.max_pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = conv2D(64, 128, 3, 1, 0)
        self.max_pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = conv2D(128, 256, 3, 1, 0)
        self.max_pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = conv2D(256, 512, 3, 1, 0)
        self.max_pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = conv2D(512, 1024, 3, 1, 0)
        
        self.upconv1 = nn.Upsample(2, 'bilinear')
        self.conv6 = conv2D(1024, 512, 3, 1, 0)

        self.upconv2 = nn.Upsample(2, 'bilinear')
        self.conv7 = conv2D(512, 256, 3, 1, 0)

        self.upconv3 = nn.Upsample(2, 'bilinear')
        self.conv8 = conv2D(256, 128, 3, 1, 0)

        self.upconv4 = nn.Upsample(2, 'bilinear')
        self.conv9 = conv2D(128, 64, 3, 1, 0)

        self.conv1_1 = nn.Conv2d(64, out_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x1)
        x2 = self.max_pool1(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool2(x2)

        x3 = self.conv3(x3)
        x4 = self.max_pool3(x3)

        x4 = self.conv4(x4)
        x5 = self.max_pool4(x4)

        x5 = self.conv5(x5)
        x5 = self.upconv1(x5)

        center_crop = f.center_crop(x4, x5.shape[2:])
        x5 = torch.cat([center_crop, x5], 1)

        x5 = self.conv6(x5)
        x5 = self.upconv2(x5)

        center_crop = f.center_crop(x3, x5.shape[2:])
        x5 = torch.cat([center_crop, x5], 1)

        x5 = self.conv7(x5)
        x5 = self.upconv3(x5)

        center_crop = f.center_crop(x2, x5.shape[2:])
        x5 = torch.cat([center_crop, x5], 1)

        x5 = self.conv8(x5)
        x5 = self.upconv4(x5)

        center_crop = f.center_crop(x1, x5.shape[2:])
        x5 = torch.cat([center_crop, x5], 1)

        x5 = self.conv9(x5)
        x5 = self.conv1_1(x5)        
        return x5

