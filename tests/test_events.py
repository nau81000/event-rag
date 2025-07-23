""" RAG unit tests
"""
import os
import pandas as pd
from utils.config import INPUT_DIR, INPUT_FILENAME


class TestMigration:
    """ Unit test class
    """
    def setup_class(self):
        """ Initialise l'environnement et le dataframe
        """
        # Chargement des données (Occitanie, date >= 2024-07-01)
        input_path = os.path.join(INPUT_DIR, INPUT_FILENAME)
        self.df_agenda = pd.read_json(input_path)

    def test_region(self):
        """ S'assurer que tous les évènements soient en Occitanie
        """
        region = self.df_agenda['location_region'].unique()
        assert len(region) == 1 and region[0] == 'Occitanie'

    def test_annee(self):
        """ S'assurer que tous les évènements aient lieu après début juillet 2024
        """
        filtered_df = self.df_agenda.loc[self.df_agenda['firstdate_begin'] >= '2024-07-01', :]
        assert (filtered_df).shape[0] == self.df_agenda.shape[0]

    def test_uid_na(self):
        """ S'assurer qu'il n'y ait pas de valeurs manquantes sur la colonne uid
        """
        assert self.df_agenda['uid'].isna().sum() == 0

    def test_uid_duplicate(self):
        """ S'assurer qu'il n'y ait pas de doublons sur la colonne uid
        """
        assert self.df_agenda['uid'].duplicated().sum() == 0
