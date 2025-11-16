""" RAG unit tests
"""
import polars as pl
from utils.config import INPUT_FILENAME


class TestMigration:
    """ Unit test class
    """
    def setup_class(self):
        """ Initialise l'environnement et le dataframe
        """
        # Chargement des données
        self.df_agenda = pl.read_json(INPUT_FILENAME)

    def test_uid_na(self):
        """ S'assurer qu'il n'y ait pas de valeurs manquantes sur la colonne uid
        """
        assert self.df_agenda.select(pl.col("uid").is_null().sum()).item() == 0

    def test_uid_duplicate(self):
        """ S'assurer qu'il n'y ait pas de doublons sur la colonne uid
        """
        assert self.df_agenda["uid"].is_duplicated().sum() == 0
