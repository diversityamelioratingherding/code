import logging

import pandas as pd
import nltk
from nltk.corpus import stopwords
from scipy.stats import entropy as scipy_entropy
from pyserini.vectorizer import TfidfVectorizer
from pyserini.index.lucene import IndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer


class CustomFeatures:
    """
    Class responsible for calculating custom features such as stopword fraction, stopword coverage,
    and entropy of term distributions in document.
    """

    def __init__(self):
        """
        Initialize the CustomFeatures class by downloading NLTK stopwords
        and setting up the stopwords set.
        """
        logging.info("Downloading NLTK stopwords...")
        nltk.download('stopwords')
        self.__stop_words = set(stopwords.words('english'))
        logging.info("Stopwords downloaded and set up successfully.")

        logging.info("Initializing Krovetz stemmer and analyzer...")
        self.__analyzer_with_stopwrods = Analyzer(get_lucene_analyzer(stemmer='krovetz', stopwords=False))
        self.__analyzer_without_stopwrods = Analyzer(get_lucene_analyzer(stemmer='krovetz', stopwords=True))
        logging.info("Analyzer initialized successfully.")
    def frac_stop(self, text_series: pd.Series) -> pd.Series:
        """
        Calculate the fraction of stopwords in each document of the text series.

        :param text_series: Series of document texts.
        :return: Series of stopword fractions for each document.
        """
        logging.info("Calculating fraction of stopwords...")

        def calculate_frac_stop(doc_text: str) -> float:
            """
            Calculate the fraction of stopwords in a single document text.

            :param doc_text: String representing the document text.
            :return: Fraction of stopwords in the document.
            """
            terms = self.__analyzer_with_stopwrods.analyze(doc_text)
            if not terms:
                return 0.0
            stopword_count = sum(1 for term in terms if term in self.__stop_words)

            return stopword_count / len(terms)

        result = text_series.apply(calculate_frac_stop)
        logging.info("Fraction of stopwords calculation completed.")

        return result

    def stop_cover(self, text_series: pd.Series) -> pd.Series:
        """
        Calculate the stopword coverage for each document in the text series.

        :param text_series: Series of document texts.
        :return: Series of stopword coverage values for each document.
        """
        logging.info("Calculating stopword coverage...")

        def calculate_stop_cover(doc_text: str) -> float:
            """
            Calculate the stopword coverage in a single document text.

            :param doc_text: String representing the document text.
            :return: Coverage of stopwords in the document.
            """
            terms_set = set(self.__analyzer_with_stopwrods.analyze(doc_text))
            if not terms_set:
                return 0.0
            stopword_count = sum(1 for stopword in self.__stop_words if stopword in terms_set)

            return stopword_count / len(self.__stop_words)

        result = text_series.apply(calculate_stop_cover)
        logging.info("Stopword coverage calculation completed.")

        return result

    def entropy(self, text_series: pd.Series) -> pd.Series:
        """
        Calculate the entropy of the term distribution for each document in the text series.

        :param text_series: Series of document texts.
        :return: Series of entropy values for each document.
        """
        logging.info("Calculating entropy of term distributions...")

        def calculate_entropy(doc_text: str) -> float:
            """
            Calculate the entropy of the term distribution in a single document text.

            :param doc_text: String representing the document text.
            :return: Entropy of the term distribution in the document.
            """
            terms = self.__analyzer_without_stopwrods.analyze(doc_text)
            if not terms:
                return 0.0
            term_counts = pd.Series(terms).value_counts(normalize=True)
            return scipy_entropy(term_counts)

        result = text_series.apply(calculate_entropy)
        logging.info("Entropy calculation completed.")

        return result

    def tf_idf(self, index_path, docnos: list) -> pd.DataFrame:
        """
        Calculate the TF-IDF vectors for the documents in the text series.

        :param index_path: Path to the Pyserini index.
        :param docnos: List of DOCNOs.

        :return: DataFrame containing the TF-IDF vectors for the documents.
        """
        logging.info("Calculating TF-IDF vectors...")

        vectorizer = TfidfVectorizer(index_path)
        tf_idf = vectorizer.get_vectors(docnos).toarray()

        # # Create a DataFrame with the TF-IDF vectors
        # tf_idf = pd.DataFrame(tf_idf, columns=[term for term in vectorizer.term_to_index.keys()])
        # tf_idf.index = docnos

        # return tf_idf
        return pd.Series(tf_idf)

    def calculate_micro_features(self, index_path, docs_df, queries, doc_dict) -> pd.DataFrame:
        """
               Calculate the micro-features documents in the text series.

               :param index_path: Path to the Pyserini index.
               :param docnos: List of DOCNOs.

               :return: DataFrame containing the micro-features for every documents.
               """
        logging.info("Calculating micro features...")
        index_reader = IndexReader(index_path)
        feature_a = ['add', 'rmv']
        feature_b = ['pw', 'npw']
        feature_c = ['query', 'stop_word', 'not']
        for fa in feature_a:
            for fb in feature_b:
                for fc in feature_c:
                    feature = fa + '-' + fb + '-' + fc
                    docs_df[feature] = -1
        def calculate_micro_feature_per_docno(docno, index_reader, queries, doc_related, fa, fb, fc):
            if doc_related == -1:
                return -1
            query = docno.split('-')[2][0:3]
            doc_text = index_reader.get_document_vector(docno)
            doc_text_winner = index_reader.get_document_vector(doc_related['prev_winner_docno'])
            doc_text_prev = index_reader.get_document_vector(doc_related['prev_docno'])
            doc_text_prev_g = [term for term in index_reader.get_document_vector(docno) for docno in doc_related['prev_docnos']]
            added_terms = [term for term in doc_text if term not in doc_text_prev]
            removed_terms = [term for term in doc_text_prev if term not in doc_text]
            if fa == 'add':
                temp_list = added_terms
            elif fa == 'rmv':
                temp_list = removed_terms
            else:
                print('error')
            if fc == 'query':
                ref = queries[query]['analyzed']
            elif fc == 'stop_word':
                ref = self.__stop_words
            elif fc == 'not':
                ref = queries[query]['analyzed'] + list(self.__stop_words)
                if fb == 'pw':
                    # return len([term for term in temp_list if term in doc_text_winner and term not in ref])
                    return len([term for term in temp_list if term in doc_text_prev_g and term not in ref])
                elif fb == 'npw':
                    # return len([term for term in temp_list if term not in doc_text_winner and term not in ref])
                    return len([term for term in temp_list if term not in doc_text_prev_g and term not in ref])
            else:
                print('error')
            if fb == 'pw':
                # return len([term for term in temp_list if term in doc_text_winner and term in ref])
                return len([term for term in temp_list if term in doc_text_prev_g and term in ref])
            elif fb == 'npw':
                # return len([term for term in temp_list if term not in doc_text_winner and term in ref])
                return len([term for term in temp_list if term not in doc_text_prev_g and term in ref])
            else:
                print('error')

        for fa in feature_a:
            for fb in feature_b:
                for fc in feature_c:
                    feature = fa + '-' + fb + '-' + fc
                    docs_df[feature] = docs_df.apply(lambda x: calculate_micro_feature_per_docno(x['docno'], index_reader, queries, doc_dict[x['docno']], fa, fb, fc), axis=1)
        return docs_df