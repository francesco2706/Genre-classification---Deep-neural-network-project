import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys

# --- CONFIGURAZIONE ---
NUM_CLASSES = 10 
BATCH_SIZE = 60
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
IMG_SIZE = 400
IN_CHANNELS = 3 
VAL_SPLIT_RATIO = 0.20 # 20% per la validazione
MODEL_SAVE_PATH = 'cnn_spectrogram_best.pth'

# --- PERCORSI LOCALI (ADATTARE) ---
train_data_dir = r'C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Dataset_Spectrogram\train'
test_data_dir = r'C:\Users\franc\Desktop\data science\PRIMO ANNO\python\final project\Dataset_Spectrogram\test'

# Definizione del DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo del dispositivo: {device}")


# --- TRASFORMAZIONI DATI ---
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- CLASSE DEL MODELLO CNN ---
class SpectrogramCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SpectrogramCNN, self).__init__()

        # --- ESTRATTORE DI FEATURE ---
        
        # Blocco 1: 400x400 -> 200x200
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # Aumenta canali a 16
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Blocco 2: 200x200 -> 100x100
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Aumenta canali a 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Blocco 3 (NUOVO): 100x100 -> 50x50
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Aumenta canali a 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ⚠️ CALCOLO FC INPUT ⚠️
        # Dimensione finale dopo 3 MaxPool su 400x400: 400 / 2 / 2 / 2 = 50x50
        FC_INPUT_SIZE = 64 * 50 * 50  # 64 canali * 50 * 50 = 160,000
        
        # --- CLASSIFICATORE (HEAD) ---
        self.classifier = nn.Sequential(
            nn.Linear(FC_INPUT_SIZE, 512), # Layer intermedio di riduzione
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout per il 50% dei neuroni
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x) # Passaggio nel nuovo blocco
        
        # Flattening
        x = x.reshape(x.shape[0], -1) 
        
        x = self.classifier(x)
        return x

# --- FUNZIONE DI VALIDAZIONE ---
def validate_epoch(model, dataloader, device):
    """Calcola l'accuratezza sul set di validazione."""
    model.eval()
    acc_metric = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            acc_metric.update(preds, targets)
    return acc_metric.compute().item()


# ====================================================================
# --- LOOP DI ESECUZIONE PRINCIPALE ---
# ====================================================================

if __name__ == '__main__':
    
    # --- 2. CARICAMENTO E SPLIT DATI ---
    
    full_train_dataset = datasets.ImageFolder(root=train_data_dir, transform=data_transforms)
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=data_transforms)
    
    # Calcola la dimensione del Validation set (20%)
    val_size = int(VAL_SPLIT_RATIO * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    
    # Suddividi il set di training
    train_dataset, validation_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Crea i DataLoaders (num_workers=0 per stabilità locale)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Dataset Totale: {len(full_train_dataset)} campioni")
    print(f"   -> Training Set: {len(train_dataset)} campioni")
    print(f"   -> Validation Set: {len(validation_dataset)} campioni")
    print(f"Dataset di Test: {len(test_dataset)} campioni")
    
    class_names = full_train_dataset.classes
    print(f"Classi rilevate: {class_names}")

    # --- 3. INIZIALIZZAZIONE ---
    model = SpectrogramCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0 # Per tracciare la migliore accuratezza di validazione

    # --- 4. TRAINING LOOP con VALIDATION e CHECKPOINT ---
    print("\n--- INIZIO TRAINING ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train() # Setta il modello in modalità training
        epoch_loss = 0.0
        
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

        # 4a. TRAINING
        for data, targets in tqdm(train_loader, desc="Training"):
            data, targets = data.to(device), targets.to(device)
            
            scores = model(data)
            loss = criterion(scores, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * data.size(0)

        total_train_loss = epoch_loss / len(train_dataset)
        
        # 4b. VALIDATION
        val_acc = validate_epoch(model, validation_loader, device)
        
        print(f"Training Loss: {total_train_loss:.4f} | Validation Accuracy: {val_acc:.4f}")
        
        # 4c. MODEL CHECKPOINTING (Salva il modello migliore)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f">>> Nuovo miglior modello salvato! Val Acc: {best_val_acc:.4f} <<<")


    # --- 5. VALUTAZIONE SUL TEST SET FINALE ---
    
    print("\n--- VALUTAZIONE SUL TEST SET ---")
    
    # Carica i pesi del modello che ha dato la migliore performance sul set di validazione
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"Caricati i pesi del modello migliore (Acc. Validation: {best_val_acc:.4f})")
    except FileNotFoundError:
        print("AVVISO: File di checkpoint non trovato, uso l'ultimo modello allenato.")


    acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
    precision = Precision(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(device)
    recall = Recall(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(device)

    model.eval() 
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            acc.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = acc.compute()
    test_precision = precision.compute()
    test_recall = recall.compute()

    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Macro Precision: {test_precision:.4f}")
    print(f"Test Macro Recall: {test_recall:.4f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test Set)')
    plt.show()
