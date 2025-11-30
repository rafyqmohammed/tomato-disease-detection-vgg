# 🍅 Tomato Disease Detection – VGG16

## 📌 Présentation du projet

Ce projet consiste à développer un système intelligent capable de **détecter automatiquement 10 maladies de la tomate** à partir d’images de feuilles.  
Il repose sur un modèle de **Transfer Learning** utilisant l’architecture **VGG16 pré-entraînée sur ImageNet**, adaptée et fine-tunée pour notre dataset.

L’objectif principal est d’obtenir un modèle performant, stable et robuste, tout en respectant une méthodologie scientifique rigoureuse :  
- Préparation complète du dataset  
- Normalisation et analyse du déséquilibre  
- Construction des DataLoaders  
- Recherche automatique d’hyperparamètres (Random Search)  
- Entraînement en deux phases  
- Évaluation détaillée du modèle final  

Ce projet s’inscrit dans un cadre académique et démontre l’efficacité du Transfer Learning pour les domaines agricoles.

## 🧹 Pré-traitement du dataset

Le dataset a été divisé en :

- **70%** : entraînement  
- **15%** : validation  
- **15%** : test  

Les images ont été normalisées dans **[0,1]** afin de stabiliser l’entraînement et de garantir la cohérence avec les modèles pré-entraînés.  
Une analyse statistique des classes a permis de calculer des **class weights**, appliqués dans la fonction de perte pour corriger le léger déséquilibre du train set.

---

## 📦 Construction des DataLoaders

Trois DataLoaders ont été créés :  
- **train_loader** (shuffle activé)  
- **val_loader**  
- **test_loader**

Ils assurent un chargement efficace des données en mini-batchs, réduisent la mémoire utilisée, et garantissent une interface structurée entre les données et le modèle.

---

## 🧪 Random Search — Optimisation des hyperparamètres

Une **Random Search** a été utilisée pour identifier les meilleurs hyperparamètres parmi :

- Learning Rate  
- Dropout  
- Taille du Fully Connected  
- Fonction d’activation (ReLU / LeakyReLU)  
- Nombre de couches à dégeler  
- Batch size  
- Optimiseur (Adam / SGD)

Chaque combinaison était testée pendant quelques epochs afin d’évaluer rapidement sa pertinence.  
Les hyperparamètres finaux retenus maximisent la **validation accuracy** et assurent un apprentissage stable.

---

## 🚀 Entraînement du modèle

L’entraînement a été réalisé en **deux phases complémentaires** :

### 🔹 Phase 1 — Warmup  
- Entraînement du classifier uniquement  
- Couches convolutionnelles gelées  
- Stabilisation progressive du modèle

### 🔹 Phase 2 — Fine-Tuning  
- Déblocage des 4 dernières couches convolutionnelles  
- Apprentissage plus profond et spécifique au domaine  
- Learning Rate réduit pour éviter une mise à jour trop brutale

Cette stratégie permet de combiner les connaissances générales d’ImageNet et l’adaptation fine au dataset des maladies de tomate.

---

## 📊 Évaluation finale

Le modèle final a été évalué sur le test set avec :

- **Accuracy globale**  
- **Matrice de confusion**  
- **Courbes d’entraînement** (accuracy + loss)  
- **Rapport de classification** (precision, recall, F1-score)  
- **Analyse qualitative des confusions**

Les résultats montrent une **performance remarquable (>98%)**, démontrant la robustesse du modèle et la qualité du pipeline.

---

## 👥 Contributeurs

- **Rim Salmoun**  
- **Zerktouni Roqia**  
- **Wissal El Idrissi**  
- **Rafyq Mohamed**

