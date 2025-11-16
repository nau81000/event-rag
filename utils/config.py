""" Utilitaires
"""
import os

INPUT_DIR = "inputs"                # Dossier pour les données sources après extraction
INPUT_FILENAME = os.path.join(INPUT_DIR, "events.json" )     # Fichier contenant les évènements

CHUNK_SIZE = 1500                   # Taille des chunks en *caractères* (vise ~512 tokens)
CHUNK_OVERLAP = 150                 # Chevauchement en *caractères*
EMBEDDING_BATCH_SIZE = 1           # Taille des lots pour l'API d'embedding

QDRANT_COLLECTION = "events_demo"
QDRANT_EMB_MODEL = "mxbai-embed-large"

# --- Configuration de la Recherche ---
SEARCH_K = 10                        # Nombre de documents à récupérer par défaut

# --- Configuration de l'Application ---
APP_TITLE = "Assistant pour la recommandation d'évènements"
