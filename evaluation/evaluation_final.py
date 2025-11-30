import sys, os
import torch
import torch.nn as nn
from torchvision import models

# === Import DataLoaders ===
sys.path.append(os.path.abspath(".."))
from model_training.Data_loader import test_loader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =============================
# 1. Device
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =============================
# 2. Rebuild SAME model as training
# =============================
def build_model():
    model = models.vgg16(weights=None)

    # IMPORTANT : classifier EXACT du training final
    model.classifier = nn.Sequential(
        nn.Linear(25088, 256),
        nn.LeakyReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 10)
    )

    return model.to(device)

model = build_model()

# === Load weights ===
model_path = "../model_training/vgg16_tomato_final.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"✔ Model loaded from {model_path}")

# =============================
# 3. Evaluate on Test Set
# =============================
correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)

        preds = outputs.argmax(1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

accuracy = correct / total
print(f"\n🔥 Test Accuracy = {accuracy:.4f}\n")


# =============================
# 4. Get class names (FIX)
# =============================
if hasattr(test_loader.dataset, "dataset"):
    classes = test_loader.dataset.dataset.classes
else:
    classes = test_loader.dataset.classes

print("✔ Classes loaded:", classes)


# =============================
# 5. Classification Report
# =============================
print("\n📄 Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=classes))


# =============================
# 6. CONFUSION MATRIX (SAVE)
# =============================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", annot=False,
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix — Tomato Diseases", fontsize=16)
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
print("✔ confusion_matrix.png saved.")


# =============================
# 7. ACCURACY CURVE
# =============================
try:
    train_acc = np.load("../model_training/train_acc_history.npy")
    val_acc = np.load("../model_training/val_acc_history.npy")

    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=300)
    plt.close()
    print("✔ accuracy_curve.png saved.")

except FileNotFoundError:
    print("⚠ Accuracy curves NOT generated (history files missing).")


# =============================
# 8. LOSS CURVE
# =============================
try:
    train_loss = np.load("../model_training/train_loss_history.npy")
    val_loss = np.load("../model_training/val_loss_history.npy")

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()
    print("✔ loss_curve.png saved.")

except FileNotFoundError:
    print("⚠ Loss curves NOT generated (history files missing).")
