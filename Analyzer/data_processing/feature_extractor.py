import logging
import os

import pandas as pd

from parsers.query_parser import QueryParser
from parsers.trec_parser import TrecParser
from feature_engineering.feature_extractor_wrapper import FeatureExtractorWrapper
from feature_engineering.custom_features import CustomFeatures
from feature_engineering.bert_scorer import BertScorer
from data_processing.data_processor import DataProcessor


class FeatureExtractor:
    """
    Class responsible for extracting features from documents and queries.
    """

    def __init__(self, final_path: str, queries_folder: str, output_path: str, index_path: str):
        """
        Initialize the FeatureExtractor.

        :param trectext_path: Path to the TREC text data.
        :param queries_folder: Path to the folder containing query XML files.
        :param output_path: Path to save the extracted features.
        :param index_path: Path to the document index.
        """
        self.__final_path = final_path
        self.__queries_folder = queries_folder
        self.__output_path = output_path
        self.__index_path = index_path

    def extract_features(self) -> None:
        """
        Extract features from the documents and queries, including custom and BERT-based features.
        """
        logging.info("Starting feature extraction process...")

        # Parse documents
        docs_df = self.__parse_documents()

        # Load and preprocess queries
        queries = self.__load_queries()

        doc_dict = self.__build_dict_for_micro(docs_df)

        # Calculate custom features
        docs_df = self.__calculate_custom_features(docs_df, queries, doc_dict)

        # Extract features using FeatureExtractorWrapper
        docs_df = self.__extract_features_with_wrapper(docs_df, queries)



        # Calculate BERT scores
        feature_matrix = self.__calculate_bert_scores(docs_df, queries)

        # Process and save the final data
        self.__process_and_save_final_data(feature_matrix)

        logging.info(f"Feature extraction completed and feature matrix saved to {self.__output_path}")

    def __parse_documents(self) -> pd.DataFrame:
        """
        Parse the TREC text data into a DataFrame.

        :return: DataFrame containing the parsed documents.
        """
        logging.info("read csv final df")
        docs_df = pd.read_csv(self.__final_path)
        docs_df['qid'] = docs_df['query_subtopic'].apply(lambda x: x.split('_')[0])
        return docs_df

    def __load_queries(self) -> dict:
        """
        Load and preprocess queries from the XML files.

        :return: Dictionary mapping query IDs to query text.
        """
        query_files = [os.path.join(self.__queries_folder, file) for file in os.listdir(self.__queries_folder) if file.endswith('.xml')]
        query_parser = QueryParser(query_files)
        queries = query_parser.query_loader()
        logging.info("Queries loaded and preprocessed successfully.")

        return queries

    def __calculate_custom_features(self, docs_df: pd.DataFrame, queries, doc_dict) -> pd.DataFrame:
        """
        Calculate custom features (stopword fraction, stopword coverage, entropy) for the documents.

        :param docs_df: DataFrame containing the documents.
        :return: DataFrame with custom features added.
        """
        custom_features = CustomFeatures()
        docs_df['frac_stop'] = custom_features.frac_stop(docs_df['doc_text'].str.lower())
        docs_df['stop_cover'] = custom_features.stop_cover(docs_df['doc_text'].str.lower())
        docs_df['entropy'] = custom_features.entropy(docs_df['doc_text'].str.lower())
        # docs_df['tfidf'] = custom_features.tf_idf(self.__index_path, docs_df['docno'].tolist())
        docs_df = custom_features.calculate_micro_features(self.__index_path, docs_df, queries, doc_dict)
        logging.info("Custom features calculated successfully.")

        return docs_df

    def __build_dict_for_micro(self, doc_df):
        doc_df_dict = {}
        for docno in list(doc_df['docno']):
            round = doc_df[doc_df['docno'] == docno].iloc[0]['round']
            query = doc_df[doc_df['docno'] == docno].iloc[0]['query_subtopic']
            user = doc_df[doc_df['docno'] == docno].iloc[0]['user']
            if round ==1:
                doc_df_dict[docno] = -1
            if (round > 1):
                prev_docno = doc_df[(doc_df['query_subtopic'] == query) & (doc_df['user'] == user) & (doc_df['round'] == round - 1)].iloc[0]['docno']
                prev_winner = doc_df[(doc_df['query_subtopic'] == query) & (doc_df['position'] == 1) & (doc_df['round'] == round - 1)].iloc[0]['docno']
                prev_second = doc_df[(doc_df['query_subtopic'] == query) & (doc_df['position'] == 2) & (doc_df['round'] == round - 1)].iloc[0]['docno']
                prev_docnos = list(doc_df[(doc_df['query_subtopic'] == query) & (doc_df['user'] != user) & (doc_df['round'] == round - 1)]['docno'])
                doc_df_dict[docno] = {'prev_docno': prev_docno, 'prev_winner_docno': prev_winner, 'prev_second_docno': prev_second, 'prev_docnos': prev_docnos}
        return doc_df_dict
    def __extract_features_with_wrapper(self, docs_df: pd.DataFrame, queries: dict) -> pd.DataFrame:
        """
        Extract additional features using the FeatureExtractorWrapper.

        :param docs_df: DataFrame containing the documents.
        :param queries: Dictionary mapping query IDs to query text.
        :return: DataFrame with extracted features.
        """
        fe_wrapper = FeatureExtractorWrapper(self.__index_path)
        info, data, _ = fe_wrapper.batch_extract(docs_df, queries)
        data['docno'] = info['docno']
        logging.info("Feature extraction completed successfully.")

        # Align docs_df to the order of 'docno' in info and merge with extracted features
        docs_df = docs_df.set_index('docno').loc[info['docno']].reset_index()
        final_data = pd.merge(data, docs_df, on='docno')

        return final_data

    def __calculate_bert_scores(self, docs_df: pd.DataFrame, queries: dict) -> pd.DataFrame:
        """
        Calculate BERT similarity scores for the document-query pairs.

        :param docs_df: DataFrame containing the documents.
        :param queries: Dictionary mapping query IDs to query text.
        :return: DataFrame with BERT scores added.
        """
        queries_to_docnos = docs_df.groupby('qid')['docno'].apply(list).to_dict()
        docs_dict = docs_df.set_index('docno')['doc_text'].to_dict()
        queries_dict = {qid: queries[str(qid)] for qid in docs_df['qid'].unique()}

        bert_scorer = BertScorer()
        bert_scores = bert_scorer.get_bert_scores(queries_to_docnos, docs_dict, queries_dict)
        docs_df['bert_score'] = docs_df['docno'].map(bert_scores)
        logging.info("BERT scores calculated successfully.")

        return docs_df

    def __process_and_save_final_data(self, feature_matrix: pd.DataFrame) -> None:
        """
        Process the extracted features and save the final DataFrame to a CSV file.

        :param feature_matrix: DataFrame containing the documents with features.
        """
        data_processor = DataProcessor()
        final_data = data_processor.process_final_data(feature_matrix)
        final_data.to_csv(self.__output_path, index=False)
        logging.info("Final data processed and saved successfully.")
