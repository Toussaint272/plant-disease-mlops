# 🌿 Détection des Maladies des Plantes - Pipeline MLOps Madagascar

Projet complet de Machine Learning Operations (MLOps) pour la détection
automatique de maladies foliaires sur **la tomate**, **la pomme de terre**
et **le maïs**.

## 📊 1. Performances du Modèle
- **Architecture** : MobileNetV3-Large (Transfer Learning)
- **Accuracy (Laboratoire - PlantVillage)** : **98.63%**
- **F1-Score Macro** : **98.79%**
- **Nombre de classes** : 10 classes

## 🔄 2. Pipeline MLOps Réalisé
1. **Versionnement des données & modèles** : DVC
2. **Suivi des expériences** : MLflow (backend SQLite)
3. **API d'inférence** : FastAPI + Uvicorn
4. **Tests automatisés** : pytest (4 tests unitaires validés)
5. **Intégration Continue (CI/CD)** : GitHub Actions (build cloud & tests)
6. **Conteneurisation** : Dockerfile
7. **Surveillance & Data Drift** : Evidently AI (rapport HTML de dérive)

## 🚀 3. Lancer l'API en local

```powershell
.venv\Scripts\Activate.ps1
uvicorn api.main:app --port 8000