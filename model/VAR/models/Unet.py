import torch.nn as nn
import torch



class Down_Up_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Down_Up_Conv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)

def crop_and_concat(upsampled, bypass):
    """
    将两个 feature map 在 H 和 W 上对齐后拼接（dim=1）
    - upsampled: 解码器上采样后的特征图 (N, C1, H1, W1)
    - bypass: 编码器传来的特征图 (N, C2, H2, W2)
    """
    h1, w1 = upsampled.shape[2], upsampled.shape[3]
    h2, w2 = bypass.shape[2], bypass.shape[3]

    # 计算差值
    delta_h = h2 - h1
    delta_w = w2 - w1

    # 对 encoder 输出进行中心裁剪
    bypass_cropped = bypass[:, :,
                     delta_h // 2: delta_h // 2 + h1,
                     delta_w // 2: delta_w // 2 + w1]

    # 拼接通道维
    return torch.cat([upsampled, bypass_cropped], dim=1)


class UNet(nn.Module):
    def __init__(self, num_classes=3):
        super(UNet, self).__init__()
        self.stage_down1=Down_Up_Conv(3, 64)
        self.stage_down2=Down_Up_Conv(64, 128)
        self.stage_down3=Down_Up_Conv(128, 256)
        self.stage_down4=Down_Up_Conv(256, 512)
        self.stage_down5=Down_Up_Conv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2,padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2,padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,padding=1)

        self.stage_up4=Down_Up_Conv(1024, 512)
        self.stage_up3=Down_Up_Conv(512, 256)
        self.stage_up2=Down_Up_Conv(256, 128)
        self.stage_up1=Down_Up_Conv(128, 64)
        self.stage_out=Down_Up_Conv(64, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        stage1 = self.stage_down1(x)
        x = self.maxpool(stage1)
        stage2 = self.stage_down2(x)
        x = self.maxpool(stage2)
        stage3 = self.stage_down3(x)
        x = self.maxpool(stage3)
        stage4 = self.stage_down4(x)
        x = self.maxpool(stage4)
        stage5 = self.stage_down5(x)

        x = self.up4(stage5)

        x = self.stage_up4(crop_and_concat(x, stage4))
        x = self.up3(x)
        x = self.stage_up3(crop_and_concat(x, stage3))
        x = self.up2(x)
        x = self.stage_up2(crop_and_concat(x, stage2))
        x = self.up1(x)
        x = self.stage_up1(crop_and_concat(x, stage1))
        out = self.stage_out(x)
        return out
