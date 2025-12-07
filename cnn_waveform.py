import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys
# ... (altre importazioni) ...

# --- CONFIGURAZIONE ---
NUM_CLASSES = 10 
BATCH_SIZE = 60 
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
IMG_SIZE = 600
IN_CHANNELS = 3 
MODEL_SAVE_PATH = r'C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\fusion_final\cnn_waveform_best.pth'

# --- PERCORSI LOCALI (ADATTARE) ---
train_waveform_dir = r'C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Dataset_Waveform\train'
test_waveform_dir = r'C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Dataset_Waveform\test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TRASFORMAZIONI DATI ---
data_transforms_waveform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- CLASSE DEL MODELLO (CON LAYER FINALE PER IL TRAINING SEPARATO) ---
class WaveformCNN(nn.Module): 
    def __init__(self, in_channels, num_classes):
        super(WaveformCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        # LAYER FINALE (DA USARE SOLO DURANTE IL TRAINING SEPARATO)
        self.fc1 = nn.Linear(16 * 150 * 150, num_classes) # 16 * 150 * 150 = 360000

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = self.pool(x) 
        x = F.relu(self.conv2(x)) 
        x = self.pool(x) 
        x = x.reshape(x.shape[0], -1) 
        x = self.fc1(x) # Chiamata al layer di classificazione
        return x

# --- INIZIALIZZAZIONE E LOOP DI TRAINING (omesso per brevità, ma usa la tua logica completa) ---
if __name__ == '__main__':
    train_dataset = datasets.ImageFolder(root=train_waveform_dir, transform=data_transforms_waveform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    
    model = WaveformCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    
    print("\n--- INIZIO TRAINING WAVEFORM ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        # ... (Logica di training e salvataggio simile a cnn_spectrogram.py) ...
        for data, targets in tqdm(train_loader, desc=f"Wave Epoch {epoch+1}"):
             data, targets = data.to(device), targets.to(device)
             optimizer.zero_grad()
             loss = criterion(model(data), targets)
             loss.backward()
             optimizer.step()
             epoch_loss += loss.item() * data.size(0)
        
        total_loss = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch+1} - Training Loss: {total_loss:.4f}")
        
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" -> Pesi Waveform salvati (Loss: {best_loss:.4f})")