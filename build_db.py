""" Création de l'index Faiss
"""
import logging
import json
import polars
from pathlib import Path
import requests
from utils.config import INPUT_DIR, INPUT_FILENAME
from utils.vector_store import VectorStoreManager
from datetime import date

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def first_of_month():
    """ Récupére le 1er du mois courant
    """
    today = date.today()

    # Début du mois courant
    first_of_this_month = today.replace(day=1)

    return str(first_of_this_month)

def get_events():
    """ Récupère les évènements depuis juillet 2024 en Occitanie
    """
    logging.info("--- Récupération des évènements ---")
    base_url = (
        "https://public.opendatasoft.com/api/explore/v2.1/catalog"
        "/datasets/evenements-publics-openagenda/exports/json"
    )
    params = {
        "select": "*",
        "where": f"firstdate_begin >= '{first_of_month()}'",
        "order_by": "firstdate_begin",
    }
    events = requests.get(base_url, params=params, timeout=30).json()
    # Création du répertoire input_directory s'il n'existe pas
    Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
    # Sauvegarde des évènements dans un fichier JSON
    with open(INPUT_FILENAME, "w", encoding="utf-8") as f_input:
        json.dump(events, f_input, ensure_ascii=False, indent=2)
    logging.info("%d évènements récupérés", len(events))
    return events

if __name__ == "__main__":
    # Récupération des évènements
    vector_store = VectorStoreManager()
    events = get_events()
    points = vector_store.build_qdrant_points(events)
    # Création de la base vectorielle Qdrant
    vector_store.build_qdrant_db(points)
