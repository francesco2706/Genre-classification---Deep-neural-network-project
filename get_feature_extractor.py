import torch
from torch import nn
from torchvision import models

def get_feature_extractor(device):
    print("Configuration of ResNet18")
    
    weights = models.ResNet18_Weights.DEFAULT 
    model = models.resnet18(weights=weights)
    
    model.fc = nn.Identity()  
    
    return model.to(device)

if __name__ == '__main__':
    test_device = torch.device("cpu")
    extractor = get_feature_extractor(test_device)

    print("ResNet18 configurated:", extractor(torch.randn(1, 3, 128, 128).to(test_device)).shape)
