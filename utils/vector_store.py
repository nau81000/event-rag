""" Outils pour indexation vectorielle
"""
import logging
import re
import json
import warnings
from typing import List, Dict, Optional
from dateutil import parser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Utilisé pour le format attendu par le splitter
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue, MatchText
from dateparser.search import search_dates
from dateutil.parser import parse
import ollama
import polars

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

from .config import (
    QDRANT_EMB_MODEL, INPUT_FILENAME,
    QDRANT_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_html(text: str):
    """ Supprime les balises HTML
    """
    return re.sub(r'\s+', ' ', BeautifulSoup(text or "", "html.parser").get_text().strip())

def extract_date_parts_from_timings(timings_str: str):
    """
    Retourne les dates, mois et années au format filtrable
    """
    try:
        timings = json.loads(timings_str)
        dates = set()
        months = set()
        years = set()
        for t in timings:
            dt = parser.isoparse(t["begin"])
            dates.add(dt.date().isoformat())              # ex: "2024-09-21"
            months.add(f"{dt.year}-{dt.month:02d}")       # ex: "2024-09"
            years.add(str(dt.year))                       # ex: "2024"
        return sorted(dates), sorted(months), sorted(years)
    except Exception as exc:
        print(f"Erreur d'extraction de date : {exc}")
        return [], [], []

def build_document(event: Dict[str, any]):
    """ Construit un document à partir des paramètres d'un évènement 
    """
    textual_columns = [
        'title_fr', 'description_fr', 'longdescription_fr', 'location_description_fr'
    ]
    text_values = []
    for col in textual_columns:
        value = event.get(col, '')
        value = '' if value is None else value
        text_values.append(value)
    text = " ".join(text_values).strip()
    ville = event.get("location_city")
    location_department = event.get("location_department")
    country = event.get("country_fr")
    title = event.get("title_fr", "")
    dates, months, years = extract_date_parts_from_timings(event["timings"])
    page_content = f"""
        Titre : {title},
        Ville : {ville},
        Département : {location_department},
        Pays: {country}
        Dates : {', '.join(dates)},
        Description : {clean_html(text)},
        Conditions : {clean_html(event.get("conditions_fr", ""))}
        """

    return Document(
        page_content=page_content,
        metadata={**event}
    )

class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""

    def __init__(self, load_events=False):
        self.event_chunks: List[Dict[str, any]] = []
        # Initialize Qdrant client
        self.qdrant = QdrantClient(host="localhost", port=6333, timeout=60.0)
        # Initialize Ollama client
        self.oclient = ollama.Client(host="localhost")
        # Load events source file in order the search to be more accurate
        if load_events:
            self.load_events()

    def load_events(self):
        """ Load events to be more accurate while searching
        """
        logging.info("Loading events")
        self.df_events = polars.read_json(INPUT_FILENAME)

    def _split_events_to_chunks(self, events: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Découpe les events en chunks avec métadonnées."""
        logging.info(
            "Découpage de %d events en chunks (taille=%d, chevauchement=%d)...",
            len(events), CHUNK_SIZE,CHUNK_OVERLAP
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len, # Important: mesure en caractères
            add_start_index=True, # Ajoute la position de début du chunk dans le document original
        )

        all_chunks = []
        event_counter = 0
        for event in events:
            langchain_doc = build_document(event)
            chunks = text_splitter.split_documents([langchain_doc])

            # Enrichit chaque chunk avec des métadonnées supplémentaires
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": event.get('uid'), # Identifiant unique du chunk
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata, # Métadonnées héritées du document (source,  ...)
                        "chunk_id_in_doc": idx, # Position du chunk dans son document d'origine
                        "start_index": chunk.metadata.get("start_index", -1) # Position début
                    }
                })
            event_counter += 1

        logging.info("Total de %d chunks créés.", len(all_chunks))
        return all_chunks

    def build_qdrant_points(self, events: List[Dict[str, any]]):
        """Construit les points Qdrant."""
        # Découper en chunks
        self.event_chunks = self._split_events_to_chunks(events)
        if not self.event_chunks:
            logging.error("Le découpage n'a produit aucun chunk. Impossible de construire l'index.")
            return
        # Générer les embeddings et les points
        chunk_counts = len(self.event_chunks)
        logging.info(
            "Génération des embeddings pour %d chunks...", chunk_counts 
        )        
        points = []
        idx = 1
        for chunk in self.event_chunks:
            logging.info(f"Traitement du chunk {idx}/{chunk_counts}")
            vector = self.oclient.embeddings(model=QDRANT_EMB_MODEL, prompt=chunk["text"])['embedding']
            points.append(PointStruct(
                id=int(chunk["id"]),
                vector=vector,
                payload={**chunk["metadata"]}
            ))
            idx += 1
        return points

    def build_qdrant_db(self, points):
        """Construit la base vectorielle Qdrant."""
        nb_points = len(points)
        dimension = len(points[0].vector)
        #  Création (ou recréation) de la collection Qdrant
        QDRANT_COLLECTION = "events_demo"
        self.qdrant.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=dimension,  # CLIP embedding dimensions
                distance=models.Distance.COSINE,
                on_disk=True,  # Store original vectors on disk
            ),
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=True,  # Keep quantized vectors in RAM
                )
            ),
            optimizers_config=models.OptimizersConfigDiff(
                max_segment_size=5_000_000, # Create larger segments for faster search
            ),
            hnsw_config=models.HnswConfigDiff(
                m=6,  # Lower m to reduce memory usage
                on_disk=False  # Keep the HNSW index graph in RAM
            ),
        )
        # Ingestion dans Qdrant
        logging.info(f"Ingestion de {nb_points} points dans '{QDRANT_COLLECTION}' (dim={dimension}).")
        def chunk_points(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i:i+n]

        batch_size = 1024
        idx = 1
        for batch in chunk_points(points, batch_size):
            logging.info(f"Ingestion: {min(idx*batch_size, nb_points)}/{nb_points}")
            self.qdrant.upsert(QDRANT_COLLECTION, points=batch, wait=False)
            idx += 1

    def search(self, query_text: str, k: int = 5, min_score: float = None) -> List[Dict[str, any]]:
        """
        Recherche les k chunks les plus pertinents pour une requête.

        Args:
            query_text: Texte de la requête
            k: Nombre de résultats à retourner
            min_score: Score minimum (entre 0 et 1) pour inclure un résultat

        Returns:
            Liste des chunks pertinents avec leurs scores
        """
        # Detect dates
        date_result = search_dates(
            query_text,
            languages=["fr"],
            settings={
                "DATE_ORDER": "DMY",   # ordre jour-mois-année
                "PREFER_DATES_FROM": "future",  # à ajuster selon ton cas
            },
        )
        date_filter_conditions = []
        if date_result:
            for _, date in date_result:
                print(date)
                date_filter_conditions.append(FieldCondition(key='timings', match=MatchText(text=date.strftime("%Y-%m"))))
        # Detect locations
        # Looking for city, department or country word and add filters if found
        words = query_text.lower().replace("?", "").split()
        location_filter_conditions = []
        for word in words:
            for col in ['location_city', 'location_department', 'country_fr']:
                res = self.df_events.filter(polars.col(col).str.to_lowercase().str.contains(word.lower()))
                if not res.is_empty():
                    location_filter_conditions.append(FieldCondition(key=col, match=MatchValue(value=word.capitalize())))
        res = self.oclient.embeddings(model=QDRANT_EMB_MODEL, prompt=query_text)
        query_filter = Filter(
            should=location_filter_conditions,
            must=date_filter_conditions
        )
        hits = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=res["embedding"],
            with_payload=True,
            limit=k,
            query_filter=query_filter
        ).points

        sorted_hits = sorted(
            hits,
            key=lambda p: parse(p.payload["firstdate_begin"])
        )

        return sorted_hits
