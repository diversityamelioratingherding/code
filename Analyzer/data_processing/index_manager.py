import logging

import pandas as pd

from pyserini.index.lucene import LuceneIndexer


class IndexManager:
    """
    Class responsible for managing the addition of new documents to an existing Lucene index.
    """

    def __init__(self, final_path: str, index_path: str):
        """
        Initialize the IndexManager.

        :param trectext_path: Path to the TREC text data.
        :param index_path: Path to the existing Lucene index.
        """
        self.__final_path = final_path
        self.__indexer_args = ["-index", index_path, "-storeDocvectors", "-storeContents"]

    def add_documents_to_index(self) -> pd.DataFrame:
        """
        Parse the TREC data and add the documents to the existing Lucene index.

        :return: DataFrame containing the parsed TREC data.
        """
        logging.info("Parsing TREC data...")
        final_df = pd.read_csv(self.__final_path)

        logging.info("Adding documents to the existing Lucene index...")
        self.__add_documents(final_df)
        logging.info("Documents added to the index successfully.")

        return final_df

    def __add_documents(self, final_df: pd.DataFrame) -> None:
        """
        Add the parsed documents from the DataFrame to the existing Lucene index.

        :param final_df: DataFrame containing the documents to be indexed.
        """
        # Initialize the LuceneIndexer with append mode to add documents to the existing index
        indexer = LuceneIndexer(append=True, args=self.__indexer_args)

        # Prepare the documents in a format suitable for the Lucene indexer
        docs = [{"id": row.docno, "contents": row.doc_text} for row in final_df.itertuples(index=False)]

        # Add the batch of documents to the index
        indexer.add_batch_dict(docs)
        indexer.close()

        logging.info("Documents added to the index successfully.")
