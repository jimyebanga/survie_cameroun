# 💍 Analyse de Survie · Primo-nuptialité au Cameroun
### Gradient Boosting Survival Analysis (GBS) — EDS 2018

---

## 🚀 Déploiement rapide sur Streamlit Cloud

### 1. Préparer votre dépôt GitHub

Créez un dépôt GitHub avec ces fichiers :
```
mon-repo/
├── app.py
├── requirements.txt
├── README.md
└── primo_nuptialite_clean.csv   ← optionnel (sinon mode démo)
```

### 2. Déployer sur Streamlit Cloud

1. Allez sur **[share.streamlit.io](https://share.streamlit.io)**
2. Connectez votre compte GitHub
3. Cliquez **"New app"**
4. Sélectionnez votre dépôt et `app.py`
5. Cliquez **"Deploy"** ✅

---

## 💻 Lancer en local

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

---

## 📁 Données

L'app fonctionne en **deux modes** :

| Mode | Description |
|------|-------------|
| **Données réelles** | Chargez `primo_nuptialite_clean.csv` via la barre latérale |
| **Démonstration** | Données synthétiques générées automatiquement |

### Format attendu du CSV

Le fichier doit contenir au minimum :

| Colonne | Type | Description |
|---------|------|-------------|
| `duree` | int | Durée avant entrée en union (en années) |
| `union` | bool | Événement observé (True) ou censuré (False) |
| `milieu` | str | Milieu de résidence (Urbain/Rural) |
| `instruction` | str | Niveau d'instruction |
| `richesse` | str | Quintile de richesse |
| `region` | str | Région du Cameroun |
| `religion` | str | Religion |

---

## ✨ Fonctionnalités

- **📈 Performances** — C-index train/test, jauge, détection du surapprentissage
- **🔍 Importance des variables** — Permutation importance + treemap
- **📉 Courbes de survie** — Prédictions pour N individus du jeu de test
- **🔮 Prédiction individuelle** — Formulaire interactif profil personnalisé
- **📋 Données & Export** — Statistiques descriptives + téléchargement CSV

---

## 📦 Dépendances principales

| Package | Version | Rôle |
|---------|---------|------|
| streamlit | ≥1.35 | Interface web |
| scikit-survival | ≥0.22 | Modèle GBS |
| plotly | ≥5.18 | Visualisations interactives |
| pandas / numpy | ≥2.0 / ≥1.24 | Manipulation des données |
| scikit-learn | ≥1.3 | Prétraitement & métriques |

---

> **Note :** `scikit-survival` nécessite une version compatible de `numpy` et `scikit-learn`. En cas de conflit, testez avec Python 3.10 ou 3.11.
