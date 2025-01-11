import logging

from tqdm import tqdm
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BertScorer:
    """
    Class responsible for calculating BERT-based similarity scores between queries and documents.
    """

    def __init__(self, model_name: str = "amberoad/bert-multilingual-passage-reranking-msmarco"):
        """
        Initialize the BertScorer with the specified tokenizer and model.

        :param model_name: Name of the BERT model to use for scoring.
        """
        logging.info("Loading tokenizer and model...")
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.__device).eval()
        logging.info("Tokenizer and model loaded successfully.")

    def get_bert_scores(self, queries_to_docnos: dict, docs_dict: dict, queries_dict: dict) -> dict:
        """
        Calculate BERT scores for a given set of queries and documents.

        :param queries_to_docnos: A dictionary mapping query IDs to lists of document IDs.
        :param docs_dict: A dictionary mapping document IDs to document texts.
        :param queries_dict: A dictionary mapping query IDs to query texts.
        :return: A dictionary mapping document IDs to BERT scores.
        """
        scores_dict = {}

        logging.info("Starting BERT score calculation...")
        for qid in tqdm(queries_to_docnos, desc="BERT", total=len(queries_to_docnos)):
            for docno in queries_to_docnos[qid]:
                query_text = queries_dict[qid]['original_query']
                doc_text = docs_dict[docno]

                # Tokenize the query and document text
                tokenized_input = self.__tokenizer.encode_plus(
                    query_text,
                    doc_text,
                    max_length=512,
                    truncation=True,
                    return_token_type_ids=True,
                    return_tensors='pt'
                )

                # Calculate the BERT score
                with torch.no_grad():
                    input_ids = tokenized_input['input_ids'].to(self.__device)
                    token_type_ids = tokenized_input['token_type_ids'].to(self.__device)
                    output = self.__model(input_ids, token_type_ids=token_type_ids, return_dict=False)[0]

                    # Handle the output for single-label or multi-label classification
                    if output.size(1) > 1:
                        score = torch.nn.functional.softmax(output, dim=1)[0, -1].item()
                    else:
                        score = output.item()

                # Store the score in the results dictionary
                scores_dict[docno] = score

        logging.info("BERT score calculation completed.")

        return scores_dict
