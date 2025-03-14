import os
import torch
from torch import nn
from torch.utils import model_zoo

# Define model_urls manually since it's no longer part of the public API
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# Import BasicBlock and Bottleneck directly
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNet(nn.Module):
    def __init__(self, block, layers, classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet18(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        try:
            # First, try loading using the model_zoo approach
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        except Exception as e:
            print(f"Warning: Original loading method failed: {e}")
            print("Trying alternative loading method...")

            # Alternative method using torchvision
            import torchvision.models as models
            pretrained_model = models.resnet18(weights="IMAGENET1K_V1")

            # Extract the features part (remove the final FC layer)
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items()
                              if not k.startswith('fc.')}

            # Load the weights
            model.load_state_dict(pretrained_dict, strict=False)
            print("Successfully loaded ResNet18 weights using alternative method")

    return model


def resnet34(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
        except Exception as e:
            import torchvision.models as models
            pretrained_model = models.resnet34(weights="IMAGENET1K_V1")
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items()
                              if not k.startswith('fc.')}
            model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        except Exception as e:
            import torchvision.models as models
            pretrained_model = models.resnet50(weights="IMAGENET1K_V1")
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items()
                              if not k.startswith('fc.')}
            model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
        except Exception as e:
            import torchvision.models as models
            pretrained_model = models.resnet101(weights="IMAGENET1K_V1")
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items()
                              if not k.startswith('fc.')}
            model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnet152(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
        except Exception as e:
            import torchvision.models as models
            pretrained_model = models.resnet152(weights="IMAGENET1K_V1")
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items()
                              if not k.startswith('fc.')}
            model.load_state_dict(pretrained_dict, strict=False)
    return model