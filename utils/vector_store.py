""" Outils pour indexation vectorielle
"""
import os
import pickle
import logging
import re
import json
from typing import List, Dict, Optional
import faiss
import numpy as np
from dateutil import parser
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Utilisé pour le format attendu par le splitter
from bs4 import BeautifulSoup

from .config import (
    MISTRAL_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE, EVENT_CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP
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
    location_name = event.get("location_name")
    ville = event.get("location_city")
    location_department = event.get("location_department")
    title = event.get("title_fr", "")
    dates, months, years = extract_date_parts_from_timings(event["timings"])
    page_content = f"""
        Titre : {title},
        Ville : {ville},
        Département : {location_department},
        Dates : {', '.join(dates)},
        Description : {clean_html(text)},
        Conditions : {clean_html(event.get("conditions_fr", ""))}
        """

    return Document(
        page_content=page_content,
        metadata={
            "uid": event.get("uid"),
            "lieu": location_name,
            "adresse": event.get("location_address"),
            "ville": str(ville).lower(),
            "departement": location_department,
            "dates": dates,
            "mois": months,
            "annees": years,
            "url": event.get("canonicalurl")
        }
    )

class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.event_chunks: List[Dict[str, any]] = []
        self.mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        """Charge l'index Faiss et les chunks si les fichiers existent."""
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(EVENT_CHUNKS_FILE):
            try:
                logging.info(
                    "Chargement de l'index Faiss depuis %s...",
                    FAISS_INDEX_FILE
                )
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                logging.info(
                    "Chargement des chunks depuis %s...",
                    EVENT_CHUNKS_FILE
                )
                with open(EVENT_CHUNKS_FILE, 'rb') as f_event:
                    self.event_chunks = pickle.load(f_event)
                logging.info(
                    "Index (%d vecteurs) et %d chunks chargés.",
                        self.index.ntotal, len(self.event_chunks)
                )
            except Exception as exc:
                logging.error(
                    "Erreur lors du chargement de l'index/chunks: %s",
                    str(exc)
                )
                self.index = None
                self.event_chunks = []
        else:
            logging.warning("Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide.")

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

    def _generate_embeddings(self, chunks: List[Dict[str, any]]) -> Optional[np.ndarray]:
        """Génère les embeddings pour une liste de chunks via l'API Mistral."""
        if not MISTRAL_API_KEY:
            logging.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
            return None
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        logging.info(
            "Génération des embeddings pour %d chunks (modèle: %s)...",
            len(chunks), EMBEDDING_MODEL
        )
        all_embeddings = []
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]

            logging.info(
                "  Traitement du lot %d/%d ({len(texts_to_embed)} chunks)",
                batch_num, total_batches
            )
            try:
                response = self.mistral_client.embeddings(
                    model=EMBEDDING_MODEL,
                    input=texts_to_embed
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except MistralAPIException as exc:
                logging.error(
                    "Erreur API Mistral lors de la génération d'embeddings (lot %d): %s",
                    batch_num, str(exc)
                )
                logging.error(
                    "  Détails: Status Code=%d, Message=%s",
                    exc.status_code, exc.message
                )
            except Exception as exc:
                logging.error(
                    "Erreur inattendue lors de la génération d'embeddings (lot %d): %s",
                    batch_num, str(exc)
                )
                 # Gérer l'erreur: ici on ajoute des vecteurs nuls pour ne pas bloquer
                num_failed = len(texts_to_embed)
                if all_embeddings: # Si on a déjà des embeddings, on prend la dimension du premier
                    dim = len(all_embeddings[0])
                else: # Sinon, on ne peut pas déterminer la dimension, on saute ce lot
                    logging.error(
                        ("Impossible de déterminer la dimension des embeddings"
                         ", saut du lot.")
                    )
                    continue
                logging.warning(
                    "Ajout de %d vecteurs nuls de dimension %d pour le lot échoué.",
                    num_failed, dim
                )
                all_embeddings.extend([np.zeros(dim, dtype='float32')] * num_failed)

            except Exception as exc:
                logging.error(
                    "Erreur inattendue lors de la génération d'embeddings (lot %d): %s",
                    batch_num, str(exc)
                )
                # Gérer comme ci-dessus
                num_failed = len(texts_to_embed)
                if all_embeddings:
                    dim = len(all_embeddings[0])
                else:
                    logging.error(
                        ("Impossible de déterminer la dimension des embeddings"
                         ", saut du lot.")
                    )
                    continue
                logging.warning(
                    "Ajout de %d vecteurs nuls de dimension %d pour le lot échoué.",
                    num_failed, dim
                )
                all_embeddings.extend([np.zeros(dim, dtype='float32')] * num_failed)


        if not all_embeddings:
            logging.error("Aucun embedding n'a pu être généré.")
            return None

        embeddings_array = np.array(all_embeddings).astype('float32')
        logging.info(
            "Embeddings générés avec succès. Shape: %d",
            embeddings_array.shape
        )
        return embeddings_array

    def build_index(self, events: List[Dict[str, any]]):
        """Construit l'index Faiss à partir des events."""
        if not events:
            logging.warning("Aucun document fourni pour construire l'index.")
            return

        # 1. Découper en chunks
        self.event_chunks = self._split_events_to_chunks(events)
        if not self.event_chunks:
            logging.error("Le découpage n'a produit aucun chunk. Impossible de construire l'index.")
            return

        # 2. Générer les embeddings
        embeddings = self._generate_embeddings(self.event_chunks)
        if embeddings is None or embeddings.shape[0] != len(self.event_chunks):
            logging.error(
                ("Problème de génération d'embeddings."
                " Le nombre d'embeddings ne correspond pas au nombre de chunks.")
            )
            # Nettoyer pour éviter un état incohérent
            self.event_chunks = []
            self.index = None
            # Supprimer les fichiers potentiellement corrompus
            if os.path.exists(FAISS_INDEX_FILE):
                os.remove(FAISS_INDEX_FILE)
            if os.path.exists(EVENT_CHUNKS_FILE):
                os.remove(EVENT_CHUNKS_FILE)
            return


        # 3. Créer l'index Faiss optimisé pour la similarité cosinus
        dimension = embeddings.shape[1]
        logging.info(
            "Création de l'index Faiss optimisé pour la similarité cosinus avec dimension %d...",
            dimension
        )

        # Normaliser les embeddings pour la similarité cosinus
        faiss.normalize_L2(embeddings)

        # Créer un index pour la similarité cosinus (IndexFlatIP = produit scalaire)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logging.info("Index Faiss créé avec %d vecteurs.", self.index.ntotal)

        # 4. Sauvegarder l'index et les chunks
        self._save_index_and_chunks()

    def _save_index_and_chunks(self):
        """Sauvegarde l'index Faiss et la liste des chunks."""
        if self.index is None or not self.event_chunks:
            logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(EVENT_CHUNKS_FILE), exist_ok=True)

        try:
            logging.info("Sauvegarde de l'index Faiss dans %s...", FAISS_INDEX_FILE)
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logging.info("Sauvegarde des chunks dans %s...", EVENT_CHUNKS_FILE)
            with open(EVENT_CHUNKS_FILE, 'wb') as f:
                pickle.dump(self.event_chunks, f)
            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as exc:
            logging.error("Erreur lors de la sauvegarde de l'index/chunks: %s", str(exc))

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
        if self.index is None or not self.event_chunks:
            logging.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not MISTRAL_API_KEY:
            logging.error(
                ("Recherche impossible: MISTRAL_API_KEY manquante"
                " pour générer l'embedding de la requête.")
            )
            return []

        logging.info(
            "Recherche des %d chunks les plus pertinents pour: '%s'",
            k, query_text
        )
        try:
            # 1. Générer l'embedding de la requête
            response = self.mistral_client.embeddings(
                model=EMBEDDING_MODEL,
                input=[query_text] # La requête doit être une liste
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')

            # Normaliser l'embedding de la requête pour la similarité cosinus
            faiss.normalize_L2(query_embedding)

            # 2. Rechercher dans l'index Faiss
            # Pour IndexFlatIP: scores = produit scalaire (plus grand = meilleur)
            # indices: index des chunks correspondants dans self.event_chunks
            # Demander plus de résultats si un score minimum est spécifié
            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            # 3. Formater les résultats
            results = []
            if indices.size > 0: # Vérifier s'il y a des résultats
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.event_chunks): # Vérifier la validité de l'index
                        chunk = self.event_chunks[idx]
                        # Convertir le score en similarité (0-1)
                        # Pour IndexFlatIP avec vecteurs normalisés, le score est déjà entre -1 et 1
                        # On le convertit en pourcentage (0-100%)
                        raw_score = float(scores[0][i])
                        similarity = raw_score * 100

                        # Filtrer les résultats en fonction du score minimum
                        # Le min_score est entre 0 et 1, mais similarity est en pourcentage (0-100)
                        min_score_percent = min_score * 100 if min_score is not None else 0
                        if min_score is not None and similarity < min_score_percent:
                            logging.debug(
                                "Document filtré (score %.2f%% < minimum %.2f%%)",
                                similarity, min_score_percent
                            )
                            continue

                        results.append({
                            "score": similarity, # Score de similarité en pourcentage
                            "raw_score": raw_score, # Score brut pour débogage
                            "text": chunk["text"],
                            "metadata": chunk["metadata"] # Contient source, category ...
                        })
                    else:
                        logging.warning(
                            "Index Faiss %d hors limites (taille des chunks: %d).",
                            idx, len(self.event_chunks)
                        )

            # Trier par score (similarité la plus élevée en premier)
            results.sort(key=lambda x: x["score"], reverse=True)
            # Limiter au nombre demandé (k) si nécessaire
            if len(results) > k:
                results = results[:k]

            if min_score is not None:
                logging.info(
                    "%d chunks pertinents trouvés (score minimum: %.2f%%).",
                    len(results), min_score * 100
                )
            else:
                logging.info("%d chunks pertinents trouvés.", len(results))

            return results

        except MistralAPIException as exc:
            logging.error(
                "Erreur API Mistral lors de la génération de l'embedding de la requête: %s",
                str(exc)
            )
            logging.error(
                "  Détails: Status Code=%d, Message=%s",
                exc.status_code, exc.message
            )
            return []
        except Exception as exc:
            logging.error(
                "Erreur inattendue lors de la recherche: %s",
                str(exc)
            )
            return []
