import sys
from .calculate import PlayerForm


def UpdatePlayerForm():
    try:
        preprocessor = PlayerForm()
        preprocessor.run()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
