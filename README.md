# Assistant RAG avec Mistral

Ce projet implémente un assistant virtuel basé sur le modèle Mistral, utilisant la technique de Retrieval-Augmented Generation (RAG) pour fournir des réponses précises et contextuelles à partir d'une base de connaissances personnalisée.

## Fonctionnalités

- 🔍 **Recherche sémantique** avec FAISS pour trouver les documents pertinents
- 🧠 **Classification des requêtes** pour déterminer si une recherche RAG est nécessaire
- ⚙️ **Paramètres personnalisables** (modèle, nombre de documents, score minimum)

## Prérequis

- Python 3.9+ 
- Clé API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))

## Installation

1. **Cloner le dépôt**

```bash
git clone <url-du-repo>
cd <nom-du-repo>
```

2. **Créer un environnement virtuel**

```bash
# Création de l'environnement virtuel
python -m venv venv

# Activation de l'environnement virtuel
# Sur Windows
venv\Scripts\activate
# Sur macOS/Linux
source venv/bin/activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Configurer la clé API**

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```
MISTRAL_API_KEY=votre_clé_api_mistral
```

## Structure du projet

```
.
├── chatbot.py              # Application Streamlit principale
├── indexer.py              # Script pour récupérer et indexer les documents
├── inputs/                 # Dossier pour les documents sources
├── vector_db/              # Dossier pour l'index FAISS et les chunks
├── utils/                  # Modules utilitaires
│   ├── config.py           # Configuration de l'application
│   └── vector_store.py     # Gestion de l'index vectoriel
```

## Utilisation

### 1. Récupérer les évènements et indexer les documents

Exécutez le script d'indexation pour récupérer, traiter les évènements et créer l'index FAISS :

```bash
python indexer.py
```
options:

**--overwrite-input** *pour écraser le fichier de données si présent*

**--no-overwrite-input** *pour ne pas écraser le fichier de données si présent (défaut)*


Ce script va :
1. Charger les évènements depuis le site Openagenda
2. Découper les évènements en chunks
3. Générer des embeddings avec Mistral
4. Créer un index FAISS pour la recherche sémantique
5. Sauvegarder l'index et les chunks dans le dossier `vector_db/`

### 3. Lancer l'application

```bash
streamlit run chatbot.py
```

L'application sera accessible à l'adresse http://localhost:8501 dans votre navigateur.

## Fonctionnalités principales

### Classification des requêtes

L'application détermine automatiquement si une question nécessite une recherche RAG ou si une réponse directe du modèle Mistral est suffisante. Cela permet d'optimiser les performances et la pertinence des réponses.

## Modules principaux

### `utils/vector_store.py`

Gère l'index vectoriel FAISS et la recherche sémantique :
- Chargement et découpage des documents
- Génération des embeddings avec Mistral
- Création et interrogation de l'index FAISS

## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `utils/config.py` :
- Chemin et nom du ficher de données
- Chemin et noms des fichiers de l'index Faiss et les chunks
- Modèles Mistral utilisés
- Taille des chunks et chevauchement
- Nombre de documents par défaut
- Nom de la commune ou organisation
