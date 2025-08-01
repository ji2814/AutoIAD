
import torch
import torch.nn as nn
import torchvision.models as models

class WiderResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(WiderResNet, self).__init__()
        self.backbone = models.wide_resnet50_2(pretrained=pretrained)

        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3
        )

    def forward(self, x):
        return self.features(x)
