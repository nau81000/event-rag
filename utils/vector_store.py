# utils/vector_store.py
import os
import pickle
import faiss
import numpy as np
import logging
import re
import json
from dateutil import parser
from typing import List, Dict, Optional
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
    except Exception as e:
        print(f"Erreur d'extraction de date : {e}")
        return [], [], []

def build_document(event: Dict[str, any]):
    """ Construit un document à partir des paramètres d'un évènement 
    """
    textual_columns = ['title_fr', 'description_fr', 'longdescription_fr', 'location_description_fr']
    text_values = []
    for col in textual_columns:
        value = event.get(col, '')
        value = '' if value is None else value
        text_values.append(value)
    text = " ".join(text_values).strip()
    #date_range = event.get("daterange_fr")
    location_name = event.get("location_name")
    ville = event.get("location_city")
    address = event.get("location_address")
    location_department = event.get("location_department")
    title = event.get("title_fr", "")
    desc = clean_html(text)
    conditions = clean_html(event.get("conditions_fr", ""))
    url = event.get("canonicalurl")
    dates, months, years = extract_date_parts_from_timings(event["timings"])
    #page_content = f"description:{title}\n{desc}\n{conditions}\nlieu:{location_name}\nadresse:{address} {ville} {location_department}\ndates:{dates} {months} {years}\nurl:{url}"
    #page_content = f"""
    #    {title} à {ville}, département {location_department}, en {months[0]}
    #    Description: {desc}
    #    Conditions : {conditions}
    #    Dates : {', '.join(dates)}
    #"""
    #print(page_content)
    page_content = f"""
        Titre : {title},
        Ville : {ville},
        Département : {location_department},
        Dates : {', '.join(dates)},
        Description : {desc},
        Conditions : {conditions}
        """

    return Document(
        page_content=page_content,
        metadata={
            "uid": event.get("uid"),
            "lieu": location_name,
            "adresse": address,
            "ville": str(ville).lower(),
            "departement": location_department,
            "dates": dates,
            "mois": months,
            "annees": years,
            "url": url
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
                logging.info(f"Chargement de l'index Faiss depuis {FAISS_INDEX_FILE}...")
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                logging.info(f"Chargement des chunks depuis {EVENT_CHUNKS_FILE}...")
                with open(EVENT_CHUNKS_FILE, 'rb') as f:
                    self.event_chunks = pickle.load(f)
                logging.info(f"Index ({self.index.ntotal} vecteurs) et {len(self.event_chunks)} chunks chargés.")
            except Exception as e:
                logging.error(f"Erreur lors du chargement de l'index/chunks: {e}")
                self.index = None
                self.event_chunks = []
        else:
            logging.warning("Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide.")

    def _split_events_to_chunks(self, events: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Découpe les events en chunks avec métadonnées."""
        logging.info(f"Découpage de {len(events)} events en chunks (taille={CHUNK_SIZE}, chevauchement={CHUNK_OVERLAP})...")
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
                            **chunk.metadata, # Métadonnées héritées du document (source, category, etc.)
                            "chunk_id_in_doc": idx, # Position du chunk dans son document d'origine
                            "start_index": chunk.metadata.get("start_index", -1) # Position de début (en caractères)
                        }
                })
            event_counter += 1

        logging.info(f"Total de {len(all_chunks)} chunks créés.")
        return all_chunks

    def _generate_embeddings(self, chunks: List[Dict[str, any]]) -> Optional[np.ndarray]:
        """Génère les embeddings pour une liste de chunks via l'API Mistral."""
        if not MISTRAL_API_KEY:
            logging.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
            return None
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        logging.info(f"Génération des embeddings pour {len(chunks)} chunks (modèle: {EMBEDDING_MODEL})...")
        all_embeddings = []
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]

            logging.info(f"  Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)")
            try:
                response = self.mistral_client.embeddings(
                    model=EMBEDDING_MODEL,
                    input=texts_to_embed
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except MistralAPIException as e:
                logging.error(f"Erreur API Mistral lors de la génération d'embeddings (lot {batch_num}): {e}")
                logging.error(f"  Détails: Status Code={e.status_code}, Message={e.message}")
            except Exception as e:
                logging.error(f"Erreur inattendue lors de la génération d'embeddings (lot {batch_num}): {e}")
                 # Gérer l'erreur: ici on ajoute des vecteurs nuls pour ne pas bloquer
                num_failed = len(texts_to_embed)
                if all_embeddings: # Si on a déjà des embeddings, on prend la dimension du premier
                    dim = len(all_embeddings[0])
                else: # Sinon, on ne peut pas déterminer la dimension, on saute ce lot
                     logging.error("Impossible de déterminer la dimension des embeddings, saut du lot.")
                     continue
                logging.warning(f"Ajout de {num_failed} vecteurs nuls de dimension {dim} pour le lot échoué.")
                all_embeddings.extend([np.zeros(dim, dtype='float32')] * num_failed)

            except Exception as e:
                logging.error(f"Erreur inattendue lors de la génération d'embeddings (lot {batch_num}): {e}")
                # Gérer comme ci-dessus
                num_failed = len(texts_to_embed)
                if all_embeddings:
                    dim = len(all_embeddings[0])
                else:
                     logging.error("Impossible de déterminer la dimension des embeddings, saut du lot.")
                     continue
                logging.warning(f"Ajout de {num_failed} vecteurs nuls de dimension {dim} pour le lot échoué.")
                all_embeddings.extend([np.zeros(dim, dtype='float32')] * num_failed)


        if not all_embeddings:
             logging.error("Aucun embedding n'a pu être généré.")
             return None

        embeddings_array = np.array(all_embeddings).astype('float32')
        logging.info(f"Embeddings générés avec succès. Shape: {embeddings_array.shape}")
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
            logging.error("Problème de génération d'embeddings. Le nombre d'embeddings ne correspond pas au nombre de chunks.")
            # Nettoyer pour éviter un état incohérent
            self.event_chunks = []
            self.index = None
            # Supprimer les fichiers potentiellement corrompus
            if os.path.exists(FAISS_INDEX_FILE): os.remove(FAISS_INDEX_FILE)
            if os.path.exists(EVENT_CHUNKS_FILE): os.remove(EVENT_CHUNKS_FILE)
            return


        # 3. Créer l'index Faiss optimisé pour la similarité cosinus
        dimension = embeddings.shape[1]
        logging.info(f"Création de l'index Faiss optimisé pour la similarité cosinus avec dimension {dimension}...")

        # Normaliser les embeddings pour la similarité cosinus
        faiss.normalize_L2(embeddings)

        # Créer un index pour la similarité cosinus (IndexFlatIP = produit scalaire)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logging.info(f"Index Faiss créé avec {self.index.ntotal} vecteurs.")

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
            logging.info(f"Sauvegarde de l'index Faiss dans {FAISS_INDEX_FILE}...")
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logging.info(f"Sauvegarde des chunks dans {EVENT_CHUNKS_FILE}...")
            with open(EVENT_CHUNKS_FILE, 'wb') as f:
                pickle.dump(self.event_chunks, f)
            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'index/chunks: {e}")

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
             logging.error("Recherche impossible: MISTRAL_API_KEY manquante pour générer l'embedding de la requête.")
             return []

        logging.info(f"Recherche des {k} chunks les plus pertinents pour: '{query_text}'")
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
                            logging.debug(f"Document filtré (score {similarity:.2f}% < minimum {min_score_percent:.2f}%)")
                            continue

                        results.append({
                            "score": similarity, # Score de similarité en pourcentage
                            "raw_score": raw_score, # Score brut pour débogage
                            "text": chunk["text"],
                            "metadata": chunk["metadata"] # Contient source, category, chunk_id_in_doc, start_index etc.
                        })
                    else:
                        logging.warning(f"Index Faiss {idx} hors limites (taille des chunks: {len(self.event_chunks)}).")

            # Trier par score (similarité la plus élevée en premier)
            results.sort(key=lambda x: x["score"], reverse=True)
            # Limiter au nombre demandé (k) si nécessaire
            if len(results) > k:
                results = results[:k]

            if min_score is not None:
                min_score_percent = min_score * 100
                logging.info(f"{len(results)} chunks pertinents trouvés (score minimum: {min_score_percent:.2f}%).")
            else:
                logging.info(f"{len(results)} chunks pertinents trouvés.")

            return results

        except MistralAPIException as e:
            logging.error(f"Erreur API Mistral lors de la génération de l'embedding de la requête: {e}")
            logging.error(f"  Détails: Status Code={e.status_code}, Message={e.message}")
            return []
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la recherche: {e}")
            return []