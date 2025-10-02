import torch.nn as nn

class InvertedBottleneck(nn.Module):
    """MobileNetV2-style block."""
    def __init__(self, in_ch, out_ch, expansion=4, stride=1):
        super().__init__()
        hid = in_ch * expansion
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU6(inplace=True),
        ) if expansion != 1 else nn.Identity()
        self.dw = nn.Sequential(
            nn.Conv2d(hid, hid, 3, stride, 1, groups=hid, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU6(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hid, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        y = self.expand(x)
        y = self.dw(y)
        y = self.project(y)
        return x + y if self.use_res else y


class MobileLiteRegressor(nn.Module):
    """Lightweight keypoint regressor with inverted bottlenecks."""
    def __init__(self, num_keypoints: int = 21):
        super().__init__()
        out_dim = num_keypoints * 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.stage = nn.Sequential(
            InvertedBottleneck(32, 32, expansion=4, stride=1),
            InvertedBottleneck(32, 48, expansion=4, stride=2),
            InvertedBottleneck(48, 48, expansion=4, stride=1),
            InvertedBottleneck(48, 64, expansion=4, stride=2),
            InvertedBottleneck(64, 64, expansion=4, stride=1),
            InvertedBottleneck(64, 96, expansion=4, stride=2),
            InvertedBottleneck(96, 128, expansion=4, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage(x)
        return self.head(x)
