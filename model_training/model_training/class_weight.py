import sys
import os

# 👉 Ajoute le dossier parent (project-tomato) au PYTHONPATH
sys.path.append(os.path.abspath(".."))

from pre_traitement_dataset.split import train_data
from collections import Counter
import torch


# 1️⃣ Récupérer toutes les classes dans train_data
train_labels = [train_data[i][1] for i in range(len(train_data))]

# 2️⃣ Compter les occurrences par classe
counter = Counter(train_labels)
print("Counts par classe :", counter)

# 3️⃣ Nombre total d'images dans train
total_images = len(train_data)

# 4️⃣ Calcul des poids inverses (classe rare = poids élevé)
class_weights = {
    cls: total_images / count
    for cls, count in counter.items()
}

print("\nClass Weights calculés :")
print(class_weights)

# 5️⃣ Convertir en Tensor (dans l'ordre des classes)
num_classes = len(class_weights)
weights_tensor = torch.zeros(num_classes)

for cls, w in class_weights.items():
    weights_tensor[cls] = w

print("\nTensor Final :")
print(weights_tensor)
