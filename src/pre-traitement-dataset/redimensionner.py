import os
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
DATASET_DIR = "/mnt/c/Users/ghout/Desktop/tomato_classification_dataset_SOKOR"
TARGET_SIZE = (224, 224)  # requis par VGG16

def resize_all_images():
    # parcourir toutes les classes directement dans DATASET_DIR
    print(f"\n📁 Traitement du dataset : {DATASET_DIR}")

    for class_name in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, class_name)

        if not os.path.isdir(class_dir):
            continue

        print(f"   🔄 Classe : {class_name}")

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(TARGET_SIZE, Image.BILINEAR)
                img.save(img_path)
            except Exception as e:
                print(f"⚠️ Erreur image {img_path} → {e}")

    print("\n✅ Toutes les images ont été redimensionnées en 224×224 !")

resize_all_images()
