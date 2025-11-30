import sys
import os
sys.path.append(os.path.abspath(".."))

from torch.utils.data import DataLoader
from pre_traitement_dataset.split import train_data, val_data, test_data

batch_size = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print("✔ DataLoaders créés avec succès")
