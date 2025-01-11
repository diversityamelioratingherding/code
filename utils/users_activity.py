import pandas as pd
from utils.general_utils import create_combine_df


def create_filters():
    "Create filters for the initial documents"
    users_activity_df = create_combine_df()
    initial_docs_df = pd.read_csv(r'data/initial_docs_final.csv')

    initial_documents_list = initial_docs_df['doc_text'].unique()

    cond = users_activity_df['doc_text'].isin(initial_documents_list)
    initial_instances = users_activity_df[cond]
    docno_set = initial_instances['docno']
    print("docno_set size:", docno_set.shape[0])

    return docno_set
