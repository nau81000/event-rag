""" Création de l'index Faiss
"""
import argparse
import logging
import json
import ijson
import io
import os
from pathlib import Path
import requests
from utils.config import INPUT_DIR, INPUT_FILENAME
from utils.vector_store import VectorStoreManager
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_events(overwrite: bool):
    """ Récupère les évènements depuis juillet 2024 en Occitanie
    """
    logging.info("--- Récupération des évènements ---")
    input_path = os.path.join(INPUT_DIR, INPUT_FILENAME)
    if os.path.exists(input_path) and not overwrite:
        # On n'écrase pas le fichier source
        return
    base_url = (
        "https://public.opendatasoft.com/api/explore/v2.1/catalog"
        "/datasets/evenements-publics-openagenda/exports/json"
    )
    cutoff = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
    params = {
        "select": "*",
        "where": f'firstdate_begin >= "{cutoff}"',
        "order_by": "location_region, location_city, firstdate_begin",
    }

    # Création du répertoire input_directory s'il n'existe pas
    Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
    # Requête en streaming pour éviter d'utiliser trop de mémoire
    with requests.get(base_url, params=params, stream=True) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True  # décompression gzip
        with open(input_path, "wb") as f_input:
            for chunk in resp.iter_content(chunk_size=8192):
                f_input.write(chunk)
    # Comptage des évènements sans tout charger en mémoire
    count = 0
    with open(input_path, "rb") as f:
        text_stream = io.TextIOWrapper(f, encoding="utf-8")
        for _ in ijson.items(text_stream, "item"):
            count += 1 
    logging.info("%d évènements récupérés", count)

def run_indexing():
    """ Exécute le processus complet d'indexation.
    """
    logging.info("--- Démarrage du processus d'indexation ---")

    input_path = os.path.join(INPUT_DIR, INPUT_FILENAME)
    with open(input_path, encoding="utf-8") as f_input:
        events = json.load(f_input)
    # --- Étape 3: Création/Mise à jour de l'index Vectoriel ---
    logging.info("Initialisation du gestionnaire de Vector Store...")
    vector_store = VectorStoreManager() # Le constructeur ne fait que charger s'il existe

    logging.info("Construction de l'index Faiss (cela peut prendre du temps)...")
    # Cette méthode va splitter, générer les embeddings, créer l'index et sauvegarder
    vector_store.build_index(events)

    logging.info("--- Processus d'indexation terminé avec succès ---")
    logging.info("Nombre d'évènements traités: %d", len(events))
    if vector_store.index:
        logging.info("Nombre de chunks indexés: %d", vector_store.index.ntotal)
    else:
        logging.warning("L'index final n'a pas pu être créé ou est vide.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'indexation pour l'application RAG")
    parser.add_argument(
        "--overwrite-input",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Écrasement du fichier input?"
    )
    args = parser.parse_args()

    # Récupération des évènements
    get_events(overwrite=args.overwrite_input)
    # Indexation des évènements
    run_indexing()
