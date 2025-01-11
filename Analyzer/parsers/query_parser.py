import logging
from typing import List, Dict
import xml.etree.ElementTree as ET

from pyserini.analysis import Analyzer, get_lucene_analyzer


class QueryParser:
    """
    Class responsible for parsing and analyzing queries from XML files using the Krovetz stemmer.
    """

    def __init__(self, files: List[str]):
        """
        Initialize the QueryParser with the given XML files and set up the analyzer.

        :param files: List of XML file paths containing the queries.
        """
        self.__files = files
        logging.info("Initializing Krovetz stemmer and analyzer...")
        self.__analyzer = Analyzer(get_lucene_analyzer(stemmer='krovetz'))
        logging.info("Analyzer initialized successfully.")

    def __parse_queries(self) -> Dict[str, Dict[str, str]]:
        """
        Parse the queries from the provided XML files and apply Krovetz stemming.

        :return: A dictionary containing the parsed queries with their details.
        """
        queries = {}

        logging.info("Starting to parse queries from XML files...")
        for file_path in self.__files:
            logging.info(f"Parsing file: {file_path}")
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                for topic in root.findall('topic'):
                    qid = topic.get('number').zfill(3)
                    query_text = topic.find('query').text.strip()
                    description = topic.find('description').text.strip()
                    subtopics = [subtopic.text.strip() for subtopic in topic.findall('subtopic')]

                    # Apply Krovetz stemming and stopword removal
                    analyzed_query = self.__analyze_query(query_text)

                    # Store the parsed query data
                    queries[qid] = {
                        'original_query': query_text,
                        'analyzed_query': ' '.join(analyzed_query),
                        'description': description,
                        'subtopics': subtopics
                    }
                    logging.debug(f"Parsed topic {qid}: {queries[qid]}")
            except ET.ParseError as e:
                logging.error(f"Error parsing {file_path}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error with {file_path}: {e}")

        logging.info("Query parsing completed.")

        return queries

    def query_loader(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load the parsed queries and add tokenized versions of the original and analyzed queries.

        :return: A dictionary containing the loaded queries with tokenized details.
        """
        logging.info("Loading and tokenizing parsed queries...")
        queries = self.__parse_queries()
        for qid, query in queries.items():
            query['analyzed'] = query['analyzed_query'].split()
            query['original'] = query['original_query'].split()
            logging.debug(f"Tokenized query {qid}: {query}")

        logging.info("Query loading and tokenizing completed.")

        return queries

    def __analyze_query(self, query_text: str) -> List[str]:
        """
        Analyze the query text using the Krovetz stemmer and return the tokenized and stemmed terms.

        :param query_text: The original query text to analyze.
        :return: A list of tokenized and stemmed terms.
        """
        return self.__analyzer.analyze(query_text)
