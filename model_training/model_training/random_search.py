import sys, os, random, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Import DataLoaders
sys.path.append(os.path.abspath(".."))
from model_training.Data_loader import train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Hyperparameters
########################################################

lr_choices = [1e-3, 5e-4, 1e-4]
optimizers = ["adam", "sgd"]
dropouts = [0.3, 0.5]
batch_sizes = [16, 32]
unfreeze_layers_choices = [0, 2, 4]
fc_sizes = [256, 512]
activations_choices = ["relu", "leakyrelu"]

NUM_EXPERIMENTS = 8
EPOCHS = 3

########################################################
# Build Model with custom params
########################################################

def build_model(dropout, unfreeze_last_layers, fc_size, activation):

    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # FREEZE all conv layers first
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze LAST conv layers
    if unfreeze_last_layers > 0:
        for param in list(model.features.parameters())[-unfreeze_last_layers:]:
            param.requires_grad = True

    # Choose activation
    if activation == "relu":
        act_fn = nn.ReLU(inplace=True)
    else:
        act_fn = nn.LeakyReLU(0.01, inplace=True)

    # Modify classifier
    model.classifier = nn.Sequential(
        nn.Linear(25088, fc_size),
        act_fn,
        nn.Dropout(dropout),
        nn.Linear(fc_size, 10)
    )

    return model.to(device)

########################################################
# Train one epoch
########################################################

def train_one_epoch(model, optimizer, loader):
    model.train()
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total

########################################################
# Validation
########################################################

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

########################################################
# Random Search
########################################################

best_acc = 0
best_params = None

for i in range(NUM_EXPERIMENTS):
    print(f"\n🔎 Random Search Versuch {i+1}/{NUM_EXPERIMENTS}")

    params = {
        "lr": random.choice(lr_choices),
        "optimizer": random.choice(optimizers),
        "dropout": random.choice(dropouts),
        "batch_size": random.choice(batch_sizes),
        "unfreeze_last_layers": random.choice(unfreeze_layers_choices),
        "fc_size": random.choice(fc_sizes),
        "activation": random.choice(activations_choices)
    }

    print("Trying:", params)

    model = build_model(
        dropout=params["dropout"],
        unfreeze_last_layers=params["unfreeze_last_layers"],
        fc_size=params["fc_size"],
        activation=params["activation"]
    )

    if params["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=params["lr"], momentum=0.9)

    # 3 EPOCHS pour meilleure estimation
    for epoch in range(EPOCHS):
        train_one_epoch(model, optimizer, train_loader)

    # Evaluate
    acc = evaluate(model, val_loader)
    print(f"➡ Validation Accuracy: {acc:.4f}")

    # Save best
    if acc > best_acc:
        best_acc = acc
        best_params = params

print("\n🎉 Best hyperparameters found:")
print(best_params)
print("🥇 Best Validation Accuracy:", best_acc)
