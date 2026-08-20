import torch
import torch.nn as nn
from torchvision import models

class DualResnet101Classifier(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, dropout_rate=0.5):
        super().__init__()

        self.resnet_primary = models.resnet101(pretrained=pretrained)
        self.resnet_lymph = models.resnet101(pretrained=pretrained)

        self.resnet_primary.fc = nn.Identity()
        self.resnet_lymph.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, primary_img, lymph_img):
        f1 = self.resnet_primary(primary_img)
        f2 = self.resnet_lymph(lymph_img)

        feat = torch.cat([f1, f2], dim=1)

        out = self.classifier(feat)
        
        return out