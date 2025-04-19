from .clean import clean_files
from .merge import merge
from .update_player_form import update_player_data
from src.playerform import UpdatePlayerForm


def preprocess_ipl_data():
    update_player_data(3)
    clean_files()
    UpdatePlayerForm()
    merge()
