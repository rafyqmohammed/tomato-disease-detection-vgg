import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = "/mnt/c/Users/ghout/Desktop/tomato_classification_dataset_SOKOR"
TARGET_SIZE = (224, 224)  # taille VGG16
SEED = 42

# Création de l'ImageDataGenerator pour augmentation + normalisation
augmenter = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

np.random.seed(SEED)

# -----------------------------
# 1. Calculer le nombre max d'images
# -----------------------------
class_counts = {}
for class_name in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    n_images = len([f for f in os.listdir(class_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    class_counts[class_name] = n_images

max_images = max(class_counts.values())
print(f"✅ Nombre max d'images pour équilibrage : {max_images}")

# -----------------------------
# 2. Augmenter chaque classe jusqu'à max_images
# -----------------------------
for class_name, count in class_counts.items():
    class_dir = os.path.join(DATASET_DIR, class_name)
    if count >= max_images:
        print(f"Classe {class_name} déjà max ({count} images), pas d'augmentation.")
        continue

    print(f"🔄 Augmentation classe {class_name} : {count} -> {max_images}")
    images = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]

    # nombre d'images à générer
    n_to_generate = max_images - count

    for i in range(n_to_generate):
        # choisir image aléatoire
        img_name = np.random.choice(images)
        img_path = os.path.join(class_dir, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(TARGET_SIZE, Image.BILINEAR)
            img_array = np.expand_dims(np.array(img), 0)  # shape (1,h,w,c)
            
            # générer une image augmentée
            aug_iter = augmenter.flow(img_array, batch_size=1)
            aug_img = next(aug_iter)[0].astype(np.uint8)

            # sauvegarder
            new_name = f"{img_name.split('.')[0]}_aug_{i}.jpg"
            new_path = os.path.join(class_dir, new_name)
            Image.fromarray(aug_img).save(new_path)

        except Exception as e:
            print(f"⚠️ Erreur image {img_path} : {e}")

print("\n✅ Augmentation terminée ! Toutes les classes ont maintenant le même nombre d'images.")
