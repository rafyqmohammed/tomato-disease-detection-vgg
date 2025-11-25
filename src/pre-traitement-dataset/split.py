from torchvision import datasets, transforms
from torch.utils.data import random_split

root = "/mnt/c/Users/ghout/Desktop/tomato_classification_dataset_SOKOR"

dataset = datasets.ImageFolder(root, transform=transforms.ToTensor())

total = len(dataset)

train_size = int(0.70 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

print("Total images :", total)
print("Train :", train_size)
print("Validation :", val_size)
print("Test :", test_size)
