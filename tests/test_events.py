import os
import sys
import pandas as pd

# Ajouter le dossier parent au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import INPUT_DIR, INPUT_FILENAME
from utils.vector_store import VectorStoreManager

class TestMigration:

    def setup_class(cls):
        """ Initialise l'environnement et le dataframe
        """
        # Chargement des données (Occitanie, date >= 2024-07-01)
        input_path = os.path.join(INPUT_DIR, INPUT_FILENAME)
        cls.df_agenda = pd.read_json(input_path)

    def test_region(self):
        """ S'assurer que tous les évènements soient en Occitanie
        """
        region = self.df_agenda['location_region'].unique()
        assert len(region) == 1 and region[0] == 'Occitanie'

    def test_annee(self):
        """ S'assurer que tous les évènements aient lieu après début juillet 2024
        """
        assert (self.df_agenda.loc[self.df_agenda['firstdate_begin'] >= '2024-07-01', :]).shape[0] == self.df_agenda.shape[0]

    def test_uid_na(self):
        """ S'assurer qu'il n'y ait pas de valeurs manquantes sur la colonne uid
        """
        assert self.df_agenda['uid'].isna().sum() == 0

    def test_uid_duplicate(self):
        """ S'assurer qu'il n'y ait pas de doublons sur la colonne uid
        """
        assert self.df_agenda['uid'].duplicated().sum() == 0
