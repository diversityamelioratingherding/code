import os
import shutil
import logging

class FileOperations:
    """
    Utility class for common file operations like copying files.
    """

    @staticmethod
    def copy_files_to_experiment_folder(trectext_path, game_history_path, experiment_folder):
        """
        Copy necessary files to the experiment folder.

        :param trectext_path: Path to the TREC text data file.
        :param game_history_path: Path to the game history file.
        :param experiment_folder: Path to the experiment folder.
        """
        # Ensure the experiment folder exists
        os.makedirs(experiment_folder, exist_ok=True)

        # Copy TREC text file
        shutil.copy(trectext_path, os.path.join(experiment_folder, os.path.basename(trectext_path)))

        # Copy game history file
        shutil.copy(game_history_path, os.path.join(experiment_folder, os.path.basename(game_history_path)))

        logging.info(f"Copied files to {experiment_folder}")
