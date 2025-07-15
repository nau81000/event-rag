import os
import sys
import pandas as pd
import faiss
import json

# Ajouter le dossier parent au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestMigration:

    def setup_class(cls):
        """ Initialise l'environnement et le dataframe
        """
        # Chargement des données (Toulouse, date >= 2024-07-01)
        cls.df_agenda = pd.read_json('openagenda.json')

    def test_region(self):
        """ S'assurer que tous les évènements soient dans le Tran
        """
        departement = self.df_agenda['location_department'].unique()
        assert len(departement) == 1 and departement[0] == 'Tarn'

    def test_annee(self):
        """ S'assurer que tous les évènements aient lieu après début juillet 2024
        """
        assert (self.df_agenda.loc[self.df_agenda['firstdate_begin'] >= '2024-07-01', :]).shape[0] == self.df_agenda.shape[0]

    def test_uid_na(self):
        """ S'assurer qu'il n'y ait pas de valeurs manquantes sur la colonne uid
        """
        assert self.df_agenda['uid'].isna().sum() == 0

    def test_all_vectors_indexed(self):
        # Charger les vecteurs originaux
        with open("vectorized_chunks.json", encoding="utf-8") as f:
            data = json.load(f)
        nb_vectors = len(data)
        # Charger l'index
        index = faiss.read_index("faiss_event_index.idx")
        indexed_vectors = index.ntotal

        assert indexed_vectors == nb_vectors, f"{indexed_vectors}/{nb_vectors} vecteurs indexés seulement !"