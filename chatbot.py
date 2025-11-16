"""Assistant RAG permettant à l'utilisateur d'interroger le système
sur des évènements culturels (Ollama + Qdrant + Streamlit).
"""

from __future__ import annotations

import json
import logging
import locale
import base64
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from streamlit_js_eval import get_geolocation

from utils.config import SEARCH_K, APP_TITLE
from utils.vector_store import VectorStoreManager
from streamlit_extras.bottom_container import bottom

# -------------------------------------------------------------------
# Configuration locale & logging
# -------------------------------------------------------------------

try:
    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
except locale.Error:
    # fallback (Windows, etc.)
    try:
        locale.setlocale(locale.LC_TIME, "fr_FR")
    except locale.Error:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -------------------------------------------------------------------
# Image helpers
# -------------------------------------------------------------------
def load_image_base64(path: str) -> str:
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# -------------------------------------------------------------------
# Geolocation helpers
# -------------------------------------------------------------------
def get_user_city() -> Optional[str]:
    """Retourne la ville de l'utilisateur via géoloc navigateur + Nominatim.

    Retourne None si la permission est refusée ou en cas d'erreur.
    """
    loc = get_geolocation()
    if not loc:
        logging.info("Géolocalisation non accordée ou indisponible.")
        return None

    lat = loc["coords"]["latitude"]
    lon = loc["coords"]["longitude"]

    headers = {
        "User-Agent": "MyEventApp/1.0 (audheon.nicolas@gmail.com)"
    }
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        address = data.get("address", {})
        # city / town / village suivant les cas
        return address.get("city") or address.get("town") or address.get("village")
    except Exception as e:
        logging.warning("Erreur lors de la géolocalisation : %s", e)
        return None


# -------------------------------------------------------------------
# Timings -> texte humain (groupés)
# -------------------------------------------------------------------

def group_timings(timings_str: str) -> List[str]:
    """Transforme un JSON de timings en une liste de plages regroupées lisibles.

    timings_str est un JSON de la forme :
    [
      {"begin": "2025-10-13T09:00:00+02:00", "end": "2025-10-13T18:00:00+02:00"},
      ...
    ]

    Retourne une liste de chaînes en français, ex :
    ["Du lundi 13 octobre 2025 au vendredi 17 octobre 2025 — 09h00 à 18h00"]
    """
    if not timings_str:
        return ["Dates non précisées"]

    try:
        timings = json.loads(timings_str)
    except Exception:
        return ["Dates non lisibles"]

    if not timings:
        return ["Dates non précisées"]

    # Convertir en objets datetime
    slots = [
        (
            datetime.fromisoformat(t["begin"]),
            datetime.fromisoformat(t["end"]),
        )
        for t in timings
    ]

    slots.sort(key=lambda x: x[0])

    groups = []
    current_start, current_end = slots[0]
    current_begin_time = current_start.strftime("%Hh%M")
    current_end_time = current_end.strftime("%Hh%M")

    for begin, end in slots[1:]:
        # Si le jour est consécutif ET même plage horaire, on regroupe
        if (
            begin.date() == current_end.date() + timedelta(days=1)
            and begin.strftime("%Hh%M") == current_begin_time
            and end.strftime("%Hh%M") == current_end_time
        ):
            current_end = end
        else:
            groups.append((current_start, current_end, current_begin_time, current_end_time))
            current_start, current_end = begin, end
            current_begin_time = begin.strftime("%Hh%M")
            current_end_time = end.strftime("%Hh%M")

    groups.append((current_start, current_end, current_begin_time, current_end_time))

    # Format humain final
    human_groups = []
    first = 0
    last = len(groups) - 1
    idx = first
    for start, end, hb, he in groups:
        start_str = start.strftime("%A %d %B %Y")
        end_str = end.strftime("%A %d %B %Y")
        if start_str == end_str:
            # un seul jour
            human_groups.append(f"Le {start_str} — {hb} à {he}")
        else:
            human_groups.append(f"Du {start_str} au {end_str} — {hb} à {he}")
        if idx!=last:
            human_groups.append(', ')
        idx += 1

    return human_groups

# -------------------------------------------------------------------
# Formatage d'un événement
# -------------------------------------------------------------------

def format_event(result: Any) -> str:
    """Formate un événement (résultat Qdrant) en Markdown lisible."""
    p: Dict[str, Any] = result.payload or {}

    title = p.get("title_fr")
    city = p.get("location_city") or "Non précisé"
    dep = p.get("location_department")
    country = p.get("location_countrycode")
    conditions = p.get("conditions_fr")
    dates = "\n    ".join(group_timings(p.get("timings")))

    return f"""- **{title}**
                \n     📍 *{city}* ({dep}, {country})
                \n     📅 **Dates :**
                    {dates}
                \n     🎟 {conditions if conditions else "Conditions d'accès non précisées"}
            """

# -------------------------------------------------------------------
# Logique RAG (Vector Store)
# -------------------------------------------------------------------

def build_response(
    vector_store_manager: VectorStoreManager,
    prompt: str
) -> str:
    """ construit le texte de réponse à partir du Vector Store.

    pour l'instant : pure retrieval + formatage.
    Si tu veux, tu pourras brancher un LLM ici plus tard.
    """
    try:
        logging.info("Recherche de contexte pour: '%s' (k=%d)", prompt, SEARCH_K)
        # À adapter si ton VectorStoreManager supporte des filtres (ville, etc.)
        search_results = vector_store_manager.search(prompt, k=SEARCH_K)
        logging.info("%d chunks trouvés.", len(search_results))
    except Exception as e:
        logging.exception("Erreur pendant vector_store_manager.search")
        return f"Une erreur est survenue pendant la recherche : {e}"

    if not search_results:
        return "Aucune information trouvée dans ma base de connaissances."

    items = [format_event(res) for res in search_results]
    return "\n".join(items)


# -------------------------------------------------------------------
# Initialisation de la session Streamlit
# -------------------------------------------------------------------

def init_session_state():
    """ Initialize state variables
    """

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"""
                    Bonjour, je suis l'assistant virtuel.
                    Comment puis-je vous aider?
                """
            }
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStoreManager(True)
    if "user_city" not in st.session_state:
        st.session_state.user_city = None

# -------------------------------------------------------------------
# UI principale Streamlit
# -------------------------------------------------------------------

def main():

    # Initialisation des états
    init_session_state()
    # Géolocalisation
    st.session_state.user_city = get_user_city()

    with st.sidebar:
        sidebar_css = f"""
        <style>
            section[data-testid="stSidebar"] {{
                background-image: url("data:image/png;base64,{load_image_base64("images/events.png")}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                color: black !important;
            }}

            section[data-testid="stSidebar"] > div:first-child {{
                background-color: rgba(180, 180, 180, 0.75);
                margin: 0px;
                border-radius: 0px;
                padding: 0px;
                text-align: center !important;
            }}

            h1 {{
                font-size: 3rem !important;
                font-weight: 700 !important;
            }}

        </style>
        """
        st.markdown(sidebar_css, unsafe_allow_html=True)
        st.title(APP_TITLE)
        st.write('Les réponses sont basées sur une base de connaissances')

    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Saisie utilisateur
    # ligne bouton + label explicatif   
    with bottom():
        if st.session_state.user_city:
            col1, col2 = st.columns([1, 6])
            with col1:
                home = st.button("❇️")
            with col2:
                user_prompt = st.chat_input("Posez votre question ou cliquer sur le bouton vert pour rechercher des évènements autour de vous…")
        else:
            home = None
            user_prompt = st.chat_input("Posez votre question…")
        st.markdown("---")
        st.caption("Powered by Ollama & Qdrant")

    query = home or user_prompt
    if query:
        if isinstance(query, bool):
            if st.session_state.user_city:
                query = f"Des évènements à {st.session_state.user_city}?"
            else:
                query = None
        # message utilisateur
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # message assistant (placeholder)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("_Recherche en cours…_")
            # Logique RAG
            response_content = build_response(
                vector_store_manager=st.session_state.vector_store,
                prompt=query
            )
            # Affichage final
            message_placeholder.markdown(response_content)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_content}
            )

    # UI: toolbar minimal + footer
    st.set_option("client.toolbarMode", "minimal")

# -------------------------------------------------------------------
# Entrée du script
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()