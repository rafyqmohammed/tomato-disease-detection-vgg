import sys, os, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np

# Import DataLoaders
sys.path.append(os.path.abspath(".."))
from model_training.Data_loader import train_loader, val_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


########################################################
# BEST HYPERPARAMETERS FROM RANDOM SEARCH
########################################################

LR = 0.001
DROPOUT = 0.3
FC_SIZE = 256
UNFREEZE_LAST_LAYERS = 4
ACTIVATION = nn.LeakyReLU(0.01, inplace=True)


########################################################
# MODEL BUILDER
########################################################

def build_model():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Freeze all conv layers initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Linear(25088, FC_SIZE),
        ACTIVATION,
        nn.Dropout(DROPOUT),
        nn.Linear(FC_SIZE, 10)
    )

    return model.to(device)


########################################################
# TRAIN + VALIDATION
########################################################

def train_epoch(model, loader, optimizer):
    model.train()
    correct, total = 0, 0
    running_loss = 0.0   # <-- ajouté pour historique

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()         # <-- ajouté
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total, running_loss / len(loader)   # <-- modifié pour retourner la loss


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0   # <-- ajouté

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            running_loss += loss.item()     # <-- ajouté

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total, running_loss / len(loader)   # <-- modifié pour retourner la loss


########################################################
# EARLY STOPPING
########################################################

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.wait = 0
        self.best_acc = 0
        self.best_model_state = None

    def check(self, acc, model):
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_model_state = model.state_dict()
            self.wait = 0
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience


########################################################
# HISTORIES (AJOUT)
########################################################

train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []


########################################################
# PHASE 1 : WARMUP
########################################################

print("\n🔵 PHASE 1 : WARMUP (FC only) — 10 epochs\n")

model = build_model()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

warmup_epochs = 10
early_stop = EarlyStopping(patience=5)

for epoch in range(warmup_epochs):
    train_acc, train_loss = train_epoch(model, train_loader, optimizer)
    val_acc, val_loss = evaluate(model, val_loader)

    print(f"[Warmup] Epoch {epoch+1}/{warmup_epochs} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

    # === SAVE HISTORY (AJOUT) ===
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    if early_stop.check(val_acc, model):
        print("⛔ Early stopping during warmup!")
        break

model.load_state_dict(early_stop.best_model_state)


########################################################
# PHASE 2 : FINE-TUNING — 40 EPOCHS
########################################################

print("\n🔵 PHASE 2 : FINE-TUNING (unfreeze last 4 conv layers) — 40 epochs\n")

params_list = list(model.features.parameters())
for param in params_list[-UNFREEZE_LAST_LAYERS:]:
    param.requires_grad = True

optimizer = optim.SGD(model.parameters(), lr=LR * 0.1, momentum=0.9)
finetune_epochs = 40
early_stop = EarlyStopping(patience=7)

for epoch in range(finetune_epochs):
    train_acc, train_loss = train_epoch(model, train_loader, optimizer)
    val_acc, val_loss = evaluate(model, val_loader)

    print(f"[FineTune] Epoch {epoch+1}/{finetune_epochs} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

    # === SAVE HISTORY (AJOUT) ===
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    if early_stop.check(val_acc, model):
        print("⛔ Early stopping during fine-tuning!")
        break

model.load_state_dict(early_stop.best_model_state)


########################################################
# SAVE MODEL
########################################################

save_path = "vgg16_tomato_final.pth"
torch.save(model.state_dict(), save_path)
print(f"\n💾 Model saved as: {save_path}")


########################################################
# SAVE HISTORY FILES (AJOUT)
########################################################


np.save("train_acc_history.npy", np.array(train_acc_history))
np.save("val_acc_history.npy", np.array(val_acc_history))
np.save("train_loss_history.npy", np.array(train_loss_history))
np.save("val_loss_history.npy", np.array(val_loss_history))

print("📁 Training histories saved!")
