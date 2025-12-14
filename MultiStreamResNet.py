import torch
from torch import nn
from get_feature_extractor import get_feature_extractor # Importa dal file 1

class MultiStreamResNet(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        
        print("Inizializzazione architettura Multi-Stream...")
        
        self.spectrogram_extractor = get_feature_extractor(device) 
        self.waveform_extractor = get_feature_extractor(device)
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, spec_data, wave_data):
        spec_features = self.spectrogram_extractor(spec_data)
        wave_features = self.waveform_extractor(wave_data)
        
        combined_features = torch.cat((spec_features, wave_features), dim=1)
        
        output = self.classifier(combined_features)

        return output
