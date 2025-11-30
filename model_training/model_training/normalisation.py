from torchvision import datasets, transforms
from torch.utils.data import random_split

root = "/mnt/c/Users/ghout/Desktop/tomato_classification_dataset_SOKOR"

# 👉 1) Dataset avec ToTensor SEULEMENT (comme tu l'as fait)
dataset = datasets.ImageFolder(root, transform=transforms.ToTensor())

# 👉 2) Split
total = len(dataset)
train_size = int(0.70 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

print("Total images :", total)
print("Train :", train_size)
print("Validation :", val_size)
print("Test :", test_size)

# --------------------------------------------------------
# ⭐ 3) Normalisation ImageNet → APPLIQUÉE APRÈS LE SPLIT
# --------------------------------------------------------

normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 👉 Appliquer la normalisation aux datasets du split
train_data.dataset.transform = normalize_transform
val_data.dataset.transform = normalize_transform
test_data.dataset.transform = normalize_transform

# --------------------------------------------------------
# ⭐ 4) Vérification de la normalisation
# --------------------------------------------------------

img, label = train_data[0]

print("\n=== Vérification Normalisation ===")
print("Shape :", img.shape)
print("Min   :", img.min().item())
print("Max   :", img.max().item())
print("Mean  :", img.mean().item())
print("Std   :", img.std().item())
