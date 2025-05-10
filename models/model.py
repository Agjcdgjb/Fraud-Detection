import torch
import torch.nn as nn
import timm

class FraudDetectionModel(nn.Module):
    def __init__(self):
        super(FraudDetectionModel, self).__init__()
        # 使用 EfficientNet-B3 作為基礎模型
        self.model = timm.create_model('efficientnet_b3', pretrained=False)
        # 修改最後一層為二分類
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
        
    def forward(self, x):
        return self.model(x) 