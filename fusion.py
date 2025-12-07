import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image
import os
import sys
from torchmetrics import Accuracy, Precision, Recall 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- CONFIGURAZIONI GLOBALI ---
NUM_CLASSES = 10
BATCH_SIZE = 60
FUSION_LR = 0.0001        # Learning rate basso per allenare solo i layer finali
FUSION_EPOCHS = 20        # Più epoche per la fusione
IN_CHANNELS = 3 
# ⚠️ I PERCORSI DEVONO ESSERE ASSOLUTI E FUNZIONANTI ⚠️
SPEC_WEIGHTS_PATH = r'C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\fusion_final\cnn_spectrogram_best.pth'
WAVE_WEIGHTS_PATH = r'C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\fusion_final\cnn_waveform_best.pth'

# --- PERCORSI DEI DATI IBRIDI ---
train_spec_dir = r'C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Dataset_Spectrogram\train'
train_wave_dir = r'C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Dataset_Waveform\train'
test_spec_dir = r'C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Dataset_Spectrogram\test'
test_wave_dir = r'C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Dataset_Waveform\test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo del dispositivo: {device}")

# ----------------------------------------------------------------------
# --- CLASSI MODIFICATE (FEATURE EXTRACTORS) ---
# ----------------------------------------------------------------------

# CNN Spectrogram (400x400 -> 160000 features)
class SpectrogramCNN_FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(SpectrogramCNN_FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.relu(self.conv1(x)) ; x = self.pool(x) 
        x = F.relu(self.conv2(x)) ; x = self.pool(x) 
        return x.reshape(x.shape[0], -1) 

# CNN Waveform (600x600 -> 360000 features)
class WaveformCNN_FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(WaveformCNN_FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.relu(self.conv1(x)) ; x = self.pool(x) 
        x = F.relu(self.conv2(x)) ; x = self.pool(x) 
        return x.reshape(x.shape[0], -1) 

# ----------------------------------------------------------------------
# --- MODELLO DI FUSIONE FINALE ---
# ----------------------------------------------------------------------

class FusionModel(nn.Module):
    def __init__(self, spec_model, wave_model, num_classes):
        super(FusionModel, self).__init__()
        
        self.spec_model = spec_model
        self.wave_model = wave_model
        
        TOTAL_FEATURE_SIZE = 160000 + 360000 # 520000
        
        # Classificatore di Fusione (nuovi layer addestrabili)
        self.intermediate_fc = nn.Linear(TOTAL_FEATURE_SIZE, 1024)
        self.classifier = nn.Linear(1024, num_classes)
        
        # Congela i layer base per il Transfer Learning
        for param in self.spec_model.parameters():
            param.requires_grad = False
        for param in self.wave_model.parameters():
            param.requires_grad = False
            
    def forward(self, spec_input, wave_input):
        spec_features = self.spec_model(spec_input)
        wave_features = self.wave_model(wave_input)
        
        combined_features = torch.cat((spec_features, wave_features), dim=1)
        
        x = F.relu(self.intermediate_fc(combined_features))
        output = self.classifier(x)
        return output

# ----------------------------------------------------------------------
# --- DATASET IBRIDO (CARICAMENTO DATI ACCOPPIATI) ---
# ----------------------------------------------------------------------

class MultiStreamDataset(Dataset):
    def __init__(self, spec_root, wave_root, spec_transforms, wave_transforms):
        # Usiamo ImageFolder come base per lo stream Spectrogram
        self.spec_dataset = datasets.ImageFolder(root=spec_root, transform=spec_transforms)
        self.wave_root = wave_root
        self.wave_transforms = wave_transforms
        self.classes = self.spec_dataset.classes

    def __len__(self):
        return len(self.spec_dataset)

    def __getitem__(self, idx):
        spec_img_tensor, label = self.spec_dataset[idx]
        
        spec_path = self.spec_dataset.imgs[idx][0]
        relative_path = os.path.relpath(spec_path, self.spec_dataset.root)
        wave_path = os.path.join(self.wave_root, relative_path)
        
        wave_img_pil = Image.open(wave_path).convert('RGB')
        wave_img_tensor = self.wave_transforms(wave_img_pil)

        return spec_img_tensor, wave_img_tensor, label 


# ----------------------------------------------------------------------
# --- FUNZIONE DI TEST E METRICHE ---
# ----------------------------------------------------------------------

def test_fusion_model(model, test_loader, device, class_names):
    from torchmetrics import Accuracy, Precision, Recall
    
    acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
    precision = Precision(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(device)
    recall = Recall(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(device)
    
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for spec_data, wave_data, labels in tqdm(test_loader, desc="Testing Fusion"):
            spec_data, wave_data, labels = spec_data.to(device), wave_data.to(device), labels.to(device)
            outputs = model(spec_data, wave_data)
            _, preds = torch.max(outputs, 1)
            
            acc.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = acc.compute()
    test_precision = precision.compute()
    test_recall = recall.compute()

    print(f"\nACCURATEZZA FINALE (FUSIONE): {test_accuracy:.4f}")
    print(f"Macro Precision (FUSIONE): {test_precision:.4f}")
    print(f"Macro Recall (FUSIONE): {test_recall:.4f}")

    print("\nClassification Report (FUSIONE):")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (FUSIONE)')
    plt.show()
    
    return test_accuracy


# ----------------------------------------------------------------------
# --- ESECUZIONE PRINCIPALE DEL TRAINING DI FUSIONE ---
# ----------------------------------------------------------------------

if __name__ == '__main__':
    
    # 1. Definizione delle Transformazioni
    data_transforms_spec = transforms.Compose([
        transforms.Resize((400, 400)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_transforms_wave = transforms.Compose([
        transforms.Resize((600, 600)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Inizializzazione dei Feature Extractor
    spec_fe = SpectrogramCNN_FeatureExtractor(in_channels=3).to(device)
    wave_fe = WaveformCNN_FeatureExtractor(in_channels=3).to(device)

    # --- CARICAMENTO PESI (CORRETTO E CRITICO) ---
    try:
        spec_state_dict = torch.load(SPEC_WEIGHTS_PATH, map_location=device)
        wave_state_dict = torch.load(WAVE_WEIGHTS_PATH, map_location=device)
        
        # Pulizia delle chiavi FC prima del caricamento (per ignorare fc1.weight/bias)
        if 'fc1.weight' in spec_state_dict:
            del spec_state_dict['fc1.weight']
            del spec_state_dict['fc1.bias']
        if 'fc1.weight' in wave_state_dict:
            del wave_state_dict['fc1.weight']
            del wave_state_dict['fc1.bias']

        # Carica solo i pesi convoluzionali rimanenti
        spec_fe.load_state_dict(spec_state_dict)
        wave_fe.load_state_dict(wave_state_dict)
        print("Pesi base caricati (layer FC scartati).")
    except Exception as e:
        print(f"ERRORE CRITICO: Impossibile caricare i pesi. Controlla che i percorsi .pth siano corretti e che i file esistano. Errore: {e}")
        sys.exit(1)

    # 3. Creazione Modello di Fusione
    fusion_model = FusionModel(spec_fe, wave_fe, num_classes=NUM_CLASSES).to(device)
    
    # 4. Data Loaders Ibridi
    fusion_train_dataset = MultiStreamDataset(train_spec_dir, train_wave_dir, data_transforms_spec, data_transforms_wave)
    fusion_train_loader = DataLoader(fusion_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    fusion_test_dataset = MultiStreamDataset(test_spec_dir, test_wave_dir, data_transforms_spec, data_transforms_wave)
    fusion_test_loader = DataLoader(fusion_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    class_names = fusion_train_dataset.classes # Ottieni i nomi delle classi

    # 5. Training Layer di Fusione
    criterion = nn.CrossEntropyLoss()
    optimizer_fusion = optim.Adam(fusion_model.parameters(), lr=FUSION_LR) 

    print(f"\n--- INIZIO TRAINING FUSIONE (Solo layer finali) per {FUSION_EPOCHS} epoche ---")
    
    for epoch in range(FUSION_EPOCHS):
        fusion_model.train()
        total_loss = 0
        
        for spec_data, wave_data, targets in tqdm(fusion_train_loader, desc=f"Fusion Epoch {epoch+1}"):
            spec_data, wave_data, targets = spec_data.to(device), wave_data.to(device), targets.to(device)
            
            optimizer_fusion.zero_grad()
            scores = fusion_model(spec_data, wave_data)
            loss = criterion(scores, targets)
            loss.backward()
            optimizer_fusion.step()
            
            total_loss += loss.item() * targets.size(0)
            
        avg_loss = total_loss / len(fusion_train_dataset)
        print(f"Fusion Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
    print("\n--- TRAINING FUSIONE COMPLETATO ---")
    
    # 6. Valutazione Finale
    print("\n--- VALUTAZIONE FINALE FUSIONE ---")
    test_fusion_model(fusion_model, fusion_test_loader, device, class_names)