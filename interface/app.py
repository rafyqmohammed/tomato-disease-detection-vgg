import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd

# -----------------------------
# Configuration de la page
# -----------------------------
st.set_page_config(
    page_title="Tomato Disease Classifier",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    /* TITRE PRINCIPAL CENTRÉ */
    .stTitle {
        color: #E74C3C;
        font-size: 3.2rem !important;
        font-weight: 700;
        text-align: center !important;
        margin-bottom: 0.3rem;
    }
    /* SOUS-TITRE AGRANDI */
    .subtitle {
        text-align: center;
        color: #7F8C8D;
        font-size: 1.6rem !important;
        font-weight: 400;
        margin-top: -10px;
        margin-bottom: 2rem;
    }
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .result-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .info-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        margin: 1rem 0;
        border-left: 4px solid #E74C3C;
    }
    .confidence-high { color: #27AE60; font-weight: bold; }
    .confidence-medium { color: #F39C12; font-weight: bold; }
    .confidence-low { color: #E74C3C; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Dictionnaire des maladies
# -----------------------------
disease_info = {
    '0_TMBS': {
        'name': 'Bacterial Spot',
        'description': 'Maladie bactérienne causant des taches sur les feuilles',
        'severity': 'Élevée',
        'treatment': 'Application de cuivre, élimination des parties infectées'
    },
    '1_TEB': {
        'name': 'Early Blight',
        'description': 'Infection fongique créant des cercles concentriques',
        'severity': 'Moyenne',
        'treatment': 'Fongicides, rotation des cultures'
    },
    '2_TLB': {
        'name': 'Late Blight',
        'description': 'Maladie fongique dévastatrice des tomates',
        'severity': 'Très élevée',
        'treatment': 'Fongicides préventifs, destruction des plants infectés'
    },
    '3_TLM': {
        'name': 'Leaf Mold',
        'description': 'Champignon affectant principalement les serres',
        'severity': 'Moyenne',
        'treatment': 'Amélioration de la ventilation, fongicides'
    },
    '4_TSLS': {
        'name': 'Septoria Leaf Spot',
        'description': 'Taches foliaires causées par un champignon',
        'severity': 'Moyenne',
        'treatment': 'Fongicides, élimination des feuilles infectées'
    },
    '5_TSM': {
        'name': 'Spider Mites',
        'description': 'Infestation par des acariens',
        'severity': 'Moyenne',
        'treatment': 'Acaricides, prédateurs naturels'
    },
    '6_TTS': {
        'name': 'Target Spot',
        'description': 'Taches circulaires sur les feuilles',
        'severity': 'Moyenne',
        'treatment': 'Fongicides, gestion de l\'humidité'
    },
    '7_TYLCV': {
        'name': 'Yellow Leaf Curl Virus',
        'description': 'Virus transmis par les aleurodes',
        'severity': 'Très élevée',
        'treatment': 'Contrôle des vecteurs, élimination des plants'
    },
    '8_TMV': {
        'name': 'Mosaic Virus',
        'description': 'Virus causant une mosaïque de couleurs',
        'severity': 'Élevée',
        'treatment': 'Élimination des plants, désinfection des outils'
    },
    '9_TH': {
        'name': 'Healthy',
        'description': 'Feuille saine sans maladie détectée',
        'severity': 'Aucune',
        'treatment': 'Maintenir les bonnes pratiques culturales'
    }
}

# -----------------------------
# Chargement du modèle
# -----------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 256),
        nn.LeakyReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 10)
    )
    model.load_state_dict(torch.load("../model_training/vgg16_tomato_final.pth", map_location=device))
    model.eval()
    return model.to(device), device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ['0_TMBS', '1_TEB', '2_TLB', '3_TLM', '4_TSLS',
           '5_TSM', '6_TTS', '7_TYLCV', '8_TMV', '9_TH']

# -----------------------------
# TITRE + SOUS-TITRE
# -----------------------------
st.markdown("""
<h1 style="
    text-align: center;
    color: #E74C3C;
    font-size: 3.2rem;
    font-weight: 700;
">
🍅 Tomato Leaf Disease Classifier
</h1>
""", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Détection intelligente des maladies de tomates par IA</p>', unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("ℹ️ À propos")
    st.write("""
    Cette application utilise un réseau de neurones profond (VGG16) 
    pour identifier les maladies des feuilles de tomates.
    """)

    # 🚨 Section Statistiques supprimée

    st.header("🎯 Comment utiliser")
    st.write("""
    1. Téléchargez une image de feuille de tomate  
    2. Attendez l'analyse automatique  
    3. Consultez les résultats et recommandations
    """)

# -----------------------------
# COLONNES
# -----------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Télécharger une image")
    uploaded_file = st.file_uploader(
        "Glissez-déposez ou sélectionnez une image",
        type=["jpg", "jpeg", "png"],
        help="Formats acceptés: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # 🔍 Vérification de la taille de l'image
        width, height = image.size
        if width > 224 or height > 224:
            st.warning("⚠️ Your image is larger than 224px — it will be resized automatically for analysis.")

        st.image(image, caption="Image téléchargée", use_container_width=True)
        
        analyze_button = st.button("🔍 Analyser l'image", type="primary", use_container_width=True)

    else:
        st.info("👆 Veuillez télécharger une image pour commencer l'analyse")
        analyze_button = False


with col2:
    if uploaded_file is not None and analyze_button:
        with st.spinner("🔄 Analyse en cours..."):
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = classes[predicted.item()]
            confidence_pct = confidence.item() * 100
            
            st.markdown("### 🎯 Résultat de l'analyse")
            
            disease = disease_info[predicted_class]

            # Couleur en fonction de la confiance
            if confidence_pct > 80:
                conf_class = "confidence-high"
                conf_emoji = "✅"
            elif confidence_pct > 60:
                conf_class = "confidence-medium"
                conf_emoji = "⚠️"
            else:
                conf_class = "confidence-low"
                conf_emoji = "❗"
            
            st.markdown(f"""
            <div class="result-box">
                <h2>{conf_emoji} {disease['name']}</h2>
                <p style="font-size: 1.5rem; margin: 1rem 0;">
                    Confiance: <span class="{conf_class}">{confidence_pct:.1f}%</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ----------- DESCRIPTION DANS LA CARTE --------------
            st.markdown(f"""
              <div class="info-card">
               <h4>📋 Description</h4>
                 <p>{disease["description"]}</p>
              </div>
            """, unsafe_allow_html=True)

            # ----------- SÉVÉRITÉ DANS LA CARTE --------------
            st.markdown(f"""
              <div class="info-card">
                <h4>⚠️ Sévérité</h4>
                 <p>{disease["severity"]}</p>
              </div>
            """, unsafe_allow_html=True)

            # ----------- TRAITEMENT DANS LA CARTE --------------
            st.markdown(f"""
              <div class="info-card">
                 <h4>💊 Traitement recommandé</h4>
                   <p>{disease["treatment"]}</p>
               </div>
            """, unsafe_allow_html=True)
            # ------------------------------------------------------

            st.markdown("### 📊 Distribution des probabilités")
            
            probs = probabilities[0].cpu().numpy() * 100
            class_names = [disease_info[c]['name'] for c in classes]
            
            sorted_indices = np.argsort(probs)[::-1][:5]
            
            chart_data = pd.DataFrame({
                'Maladie': [class_names[i] for i in sorted_indices],
                'Probabilité (%)': [probs[i] for i in sorted_indices]
            }).set_index('Maladie')
            
            st.bar_chart(chart_data, color="#E74C3C", height=350)

            with st.expander("📋 Voir toutes les probabilités"):
                full_data = pd.DataFrame({
                    'Maladie': class_names,
                    'Probabilité (%)': [f'{p:.2f}%' for p in probs]
                }).sort_values('Probabilité (%)', ascending=False)
                st.dataframe(full_data, hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7F8C8D;'>
    <p>🌱 Développé pour l'agriculture intelligente</p>
    <p style='font-size: 0.9rem;'>Modèle VGG16 entraîné sur des images de feuilles de tomates</p>
</div>
""", unsafe_allow_html=True)
