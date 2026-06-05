# Furniture Classification and Detection System

## Membres du groupe
Omomene IWELOMEN - Salma BENTISSE - Jennifer HOUNGBEDJI  
Dépôt GitHub : https://github.com/Omomene/NeuralNetwork.git

---

## Présentation du projet

Ce projet est un système complet de machine learning et de deep learning pour l’analyse de meubles utilisant le dataset IKEA et un dataset de détection YOLOv8.

Le système combine :

- Machine Learning classique (baseline Logistic Regression)
- Deep Learning (CNN personnalisé)
- Transfer Learning (MobileNetV2)
- Détection d’objets (YOLOv8)

Un dashboard Streamlit a été développé pour comparer tous les modèles en temps réel et tester les prédictions de manière interactive.

![alt text](<images/screenshots/2026-06-05 10_06_33-Greenshot.png>)

---

## Objectifs

- Construire un modèle ML de base pour la classification
- Concevoir et entraîner un CNN personnalisé from scratch
- Améliorer les performances avec le transfer learning (MobileNetV2)
- Entraîner un modèle de détection d’objets avec YOLOv8
- Comparer toutes les approches de manière équitable
- Déployer une application Streamlit en temps réel

---

## Vue d’ensemble des datasets

### Dataset de classification IKEA
- Source : IKEA product dataset - https://www.kaggle.com/datasets/ahmedkallam/ikea-sa-furniture-web-scraping
- Tâche : Classification d’images
- Classes : 4
  - Beds
  - Cabinets & cupboards
  - Sofas & armchairs
  - Tables & desks

### Dataset de détection YOLOv8 Furniture
- Source : Roboflow (Furniture Detection v20) - https://universe.roboflow.com/mokhamed-nagy-u69zl/furniture-detection-qiufc/dataset/20
- Tâche : Détection d’objets
- Images : 8,055
- Classes : 25 catégories de meubles

### Split du dataset (YOLOv8)
- Train : 6,424 images
- Validation : 891 images
- Test : 739 images

---

## Architectures des modèles

### 1. Logistic Regression (ML baseline)
- Entrée : images aplaties
- Régularisation : Ridge (L2)
- Objectif : baseline classique de comparaison

### 2. CNN personnalisé
- Couches convolutionnelles
- MaxPooling
- Régularisation Dropout
- Sortie Softmax (4 classes)

### 3. MobileNetV2 (Transfer Learning)
- Pré-entraîné sur ImageNet
- Backbone gelé
- Tête de classification personnalisée

### 4. YOLOv8 (Détection d’objets)
- YOLOv8n pré-entraîné
- Fine-tuning sur dataset meuble
- Détection par bounding boxes

---

## Résultats de performance

### Modèles de classification

| Modèle | Type | Accuracy |
|------|------|----------|
| Logistic Regression | ML baseline | 0.59 |
| CNN personnalisé | Deep Learning | 0.42 |
| MobileNetV2 | Transfer Learning | 0.62 |

### Détection d’objets (YOLOv8)

| Métrique | Valeur |
|------|--------|
| Precision | 0.43 |
| Recall | 0.37 |
| mAP50 | 0.41 |
| mAP50-95 | 0.24 |

---

## Détails d’entraînement

### CNN & MobileNetV2
- Taille des images : 224x224
- Optimiseur : Adam
- Loss : Sparse categorical crossentropy
- Epochs : 10

### YOLOv8
- Modèle : yolov8n
- Taille des images : 416x416
- Epochs : 5  
- Framework : Ultralytics

---

## Structure du projet

```


├── main.ipynb
├── src/
│ ├── data.py
│ ├── eda.py
│ ├── ml.py
│ ├── eval_ml.py
│ ├── dl.py
│ ├── opti_dl.py
│ ├── eval_dl.py
│ ├── yolo_dl.py
│
├── dataset/
│ ├── IKEA_SA_Furniture_Web_Scrapings_sss.csv
│ ├── furniture_detection/
│
├── images/
│ └── furniture_images/
│
├── runs/
│ └── detect/
│ └── furniture_detector-2/
│ ├── weights/
│ │ ├── best.pt
│ │ └── last.pt
│
├── models/
│ ├── furniture_cnn.keras
│ ├── mobilenet_model.keras
│
├── app.py
├── requirements.txt
└── README.md

````

---

## Dashboard Streamlit

L’application fournit :

### 1. Interface de prédiction
- Upload d’image
- Comparaison des prédictions :
  - CNN personnalisé
  - MobileNetV2
  - Détection YOLOv8

![alt text](<images/screenshots/2026-06-05 14_46_21-Greenshot.png>)

### 2. Analyse du dataset
- Distribution des classes
- Visualisation du split des datasets
- Vue statistique

![alt text](<images/screenshots/2026-06-05 16_21_20-Greenshot.png>)

### 3. Comparaison des modèles
- Comparaison des accuracies
- Performance YOLO (mAP)
- Graphiques

![alt text](<images/screenshots/2026-06-05 16_22_08-Greenshot.png>)

---

## Comment exécuter le projet

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
````

---

### 2. Lancer l’entraînement

```bash
python main.ipynb
```

---

### 3. Lancer le dashboard Streamlit

```bash
streamlit run app.py
```

---

## Résumé des résultats

* YOLOv8 détecte correctement les objets avec une précision moyenne
* CNN et MobileNetV2 ont des performances limitées par la capacité d’entraînement
* Logistic Regression fournit une baseline mais insuffisante pour la classification fine des meubles

---

## Reproductibilité

Tous les modèles peuvent être réentraînés via :

* Scripts modulaires dans `src/`
* Pipeline `main.ipynb`
* Structure du dataset dans `dataset/`
* Environnement `requirements.txt`

---

## Notes

* Le dataset n’est pas versionné à cause de sa taille
* Les outputs YOLO sont stockés dans `runs/`
* Les modèles sont sauvegardés en `.keras` et `.pt`



