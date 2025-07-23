""" Création de l'index Faiss
"""
import argparse
import logging
import json
import os
from pathlib import Path
import requests
from utils.config import INPUT_DIR, INPUT_FILENAME
from utils.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_events(overwrite: bool):
    """ Récupère les évènements depuis juillet 2024 en Occitanie
    """
    logging.info("--- Récupération des évènements ---")
    input_path = os.path.join(INPUT_DIR, INPUT_FILENAME)
    if os.path.exists(input_path) and not overwrite:
        # On n'écrase pas le fichier source
        return
    records = []
    limit = 100
    offset = 0
    uids = []
    base_url = (
        "https://public.opendatasoft.com/api/explore/v2.1/catalog"
        "/datasets/evenements-publics-openagenda/records"
    )
    while True:
        params = {
            "select": "*",
            "where": 'firstdate_begin >= "2024-07-01"',
            "limit": limit,
            "offset": offset,
            "order_by": "location_city, firstdate_begin",
            "refine": 'location_region:"Occitanie"',
        }
        resp = requests.get(base_url, params=params, timeout=30).json()
        results = resp.get('results', [])
        for event in results:
            uid = event.get("uid")
            if uid not in uids:
                records.append(event)
                # On garde trace des uid déjà vus pour ne les avoir qu'une fois
                uids.append(uid)
        if len(results) < limit:
            break
        offset += 1
    # Création du répertoire input_directory s'il n'existe pas
    Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
    # Sauvegarde des évènements dans un fichier JSON
    with open(input_path, "w", encoding="utf-8") as f_input:
        json.dump(records, f_input, ensure_ascii=False, indent=2)
    logging.info("%d évènements récupérés", len(records))

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
