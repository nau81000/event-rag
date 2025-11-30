# Assistant RAG

Ce projet implémente un assistant virtuel basé sur le modèle Ollama, utilisant la technique de Retrieval-Augmented Generation (RAG) pour fournir des réponses précises et contextuelles à partir d'une base de connaissances personnalisée.

## Fonctionnalités

- 🔍 **Recherche sémantique** avec Ollama pour trouver les documents pertinents
- 🧠 **Classification des requêtes** pour déterminer si une recherche RAG est nécessaire
- ⚙️ **Paramètres personnalisables** (modèle, nombre de documents, score minimum)

## Prérequis

- Python 3.12+
- Docker
- Npm

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

```bash
cd frontend
npm install
npm run build
```

4. **Prérequis**

Avoir installé Ollama sur la machine hôte. L'utilisation d'Ollama dans un container Docker est déconseillé car la création des embeddings sera très longue, Docker ne gérant pas l'accélération GPU.

Télécharger le modèle d'embeddings:

```bash
ollama pull mxbai-embed-large
```

## Structure du projet

```
├── .github/workflows/test.yml  # Workflow pour lancement automatique des tests unitaires sur github
├── build_db.py                 # Script pour récupérer et construire la base vectorielle
├── frontend/                   # Dossier comprenant les fichiers de l'interface utilisateur basée sur React
├── inputs/                     # Dossier pour les documents sources
├── pytest.ini                  # Fichier d'initialisation de pytest
├── README.md                   # README du projet 
├── server.py                   # Script de lancement du serveur web
├── tests/test_events.py        # Tests unitaires
├── utils/                      # Modules utilitaires
│   ├── config.py               # Configuration de l'application
│   └── rag.py                  # API REST
│   └── vector_store.py         # Gestion de l'index vectoriel
```

## Utilisation

### 1. Récupérer les évènements et indexer les documents

Créer la base vectorielle Qdrant

```bash
docker-compose up -d
```

Exécuter le script d'indexation pour récupérer, traiter les évènements et alimenter la base vectorielle :

```bash
python build_db.py
```

Ce script va :
1. Charger les évènements depuis le site Openagenda
2. Découper les évènements en chunks
3. Générer des embeddings avec Ollama
4. Créer une base vectorielle avec Qdrant (dans un container Docker)


### 2. Lancer l'application

```bash
python server.py
```

L'application sera accessible à l'adresse http://localhost:8000 dans votre navigateur.

## Fonctionnalités principales

### Classification des requêtes

Les requêtes sont analysés et optimiser par l'utilisation de filtres sur les lieux et dates. 

## Modules principaux

### `utils/vector_store.py`

Gère la base vectorielle et la recherche sémantique :
- Chargement et découpage des documents
- Génération des embeddings avec Ollama
- Création et interrogation de la base Qdrant

## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `utils/config.py` :
- Chemin et nom du ficher de données
- Taille des chunks et chevauchement
- Nombre de documents par défaut

## Lancement des tests unitaires

```bash
pytest tests
```

## Démonstration
<video controls>
  <source src="demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>