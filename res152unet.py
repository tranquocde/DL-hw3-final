import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class DecBlock(nn.Module):

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 ):
        super().__init__()

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels
        self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        x = self.upsample(up_x) #(-1,1024,-,-)
        x = torch.cat([x, down_x], 1) #(-1,2048,-,-) # channels nhân đôi lên
        x = self.conv_block_1(x)#(-1,1024,-,-)
        x = self.conv_block_2(x)
        return x


class Resnet152Unet(nn.Module):
    # DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        resnet = resnet152(weights=ResNet152_Weights.DEFAULT)
        blocks = list(resnet.children())
        enc_blocks = []
        self.input_block= nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.input_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        for enc in list(blocks):
            if isinstance(enc, nn.Sequential):
                enc_blocks.append(enc)
        self.enc_blocks = nn.ModuleList(enc_blocks)
        
        self.bottleneck = BottleNeck(2048, 2048)

        dec_blocks = [
            DecBlock(2048, 1024),
            DecBlock(1024, 512),
            DecBlock(512, 256),
            DecBlock(in_channels=128 + 64, out_channels=128, up_conv_in_channels=256, up_conv_out_channels=128),
            DecBlock(in_channels=64 + 3, out_channels=64, up_conv_in_channels=128, up_conv_out_channels=64)
        ]

        self.dec_blocks = nn.ModuleList(dec_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        pre_skip = dict() # to save skip-connection

        pre_skip["layer0"] = x
        x = self.input_block(x) # (-1,64,112,112)
        pre_skip["layer1"] = x
        x = self.input_pool(x) #-1,64,56,56

        for i, block in enumerate(self.enc_blocks, 2):
            x = block(x)
            if i == 5 :
                continue
            pre_skip[f"layer{i}"] = x # save the skip-connection

        x = self.bottleneck(x) #(-1,2048,224,224)

        for i, block in enumerate(self.dec_blocks, 1):

            key = f"layer{5 - i}"
            x = block(x, pre_skip[key]) # decode with x and saved skip-connection

        x = self.out(x)
        del pre_skip
        return x