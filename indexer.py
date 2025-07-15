""" Création de l'index Faiss
"""
import argparse
import logging
import requests
import json
import os
from pathlib import Path
from utils.config import INPUT_DIR, INPUT_FILENAME
from utils.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_events(input_directory: str, input_filename: str, overwrite: bool):
    """ Récupère les évènements depuis juillet 2024 dans le Tarn
    """
    input_path = os.path.join(input_directory, input_filename)
    if os.path.exists(input_path) and not overwrite:
        # On n'écrase pas le fichier source
        return
    records = []
    limit = 100
    offset = 0
    while True:
        url = f"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records?select=*&where=firstdate_begin%20%3E%3D%20%222024-07-01%22&limit={limit}&offset={offset}&refine=location_department%3A%22Tarn%22"
        resp = requests.get(url, params={}).json()
        results = resp.get('results', [])
        records.extend(results)
        if len(results) < limit:
            break
        offset += 1
    # Création du répertoire input_directory s'il n'existe pas
    Path(input_directory).mkdir(parents=True, exist_ok=True)
    # Sauvegarde des évènements dans un fichier JSON
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def run_indexing(input_directory: str, input_filename: str):
    """ Exécute le processus complet d'indexation.
    """
    logging.info("--- Démarrage du processus d'indexation ---")

    input_path = os.path.join(input_directory, input_filename)
    with open(input_path, encoding="utf-8") as f:
        events = json.load(f)
    # --- Étape 3: Création/Mise à jour de l'index Vectoriel ---
    logging.info("Initialisation du gestionnaire de Vector Store...")
    vector_store = VectorStoreManager() # Le constructeur ne fait que charger s'il existe

    logging.info("Construction de l'index Faiss (cela peut prendre du temps)...")
    # Cette méthode va splitter, générer les embeddings, créer l'index et sauvegarder
    vector_store.build_index(events)

    logging.info("--- Processus d'indexation terminé avec succès ---")
    logging.info(f"Nombre d'évènements traités: {len(events)}")
    if vector_store.index:
        logging.info(f"Nombre de chunks indexés: {vector_store.index.ntotal}")
    else:
        logging.warning("L'index final n'a pas pu être créé ou est vide.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'indexation pour l'application RAG")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Répertoire contenant le fichier source (par défaut: {INPUT_DIR})"
    )
    parser.add_argument(
        "--input-filename",
        type=str,
        default=INPUT_FILENAME,
        help=f"Nom du fichier source (par défaut: {INPUT_DIR})"
    )
    parser.add_argument(
        "--overwrite-input",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=f"Écrasement du fichier input?"
    )
    args = parser.parse_args()

    # Récupération des évènements
    get_events(input_directory=args.input_dir, input_filename=args.input_filename, overwrite=args.overwrite_input)
    # Indexation des évènements
    run_indexing(input_directory=args.input_dir, input_filename=args.input_filename)