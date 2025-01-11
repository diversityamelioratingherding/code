import os
import hashlib
import datetime
from data_processing.index_manager import IndexManager
from data_processing.feature_extractor import FeatureExtractor
from analysis.analyzer import Analyzer
from utils.file_operations import FileOperations
from utils.logging_setup import LoggingSetup


def main():
    # LoggingSetup.configure_logging("experiment.log")
    # Define paths and parameters
    final_data = 'Analyzer/data/final_df.csv'
    queries_folder = 'Analyzer/data/web_track'
    feature_matrix_path = 'Analyzer/data/output/feature_matrix.csv'
    index_path = '<PATH_TO_CLUEWEB09_PYSERINI>'

    # Add documents to the background index
    # index_manager = IndexManager(final_data, index_path)
    # index_manager.add_documents_to_index()

    # Extract features
    # feature_extractor = FeatureExtractor(final_data, queries_folder, feature_matrix_path, index_path)
    # feature_extractor.extract_features()


if __name__ == "__main__":
    main()
