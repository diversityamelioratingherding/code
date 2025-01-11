import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

NUM_ROUNDS = 7
MAX_RANK = 4

E5_REPRESENTATION = 'e5_representation'
TF_IDF_REPRESENTATION = 'tfidf-krovetz'
JACCARD_REPRESENTATION = 'tfidf-krovetz-jaccard'
BERT_REPRESENTATION = 'bert_representation'
E5_REPRESENTATION_PATH = fr'data/{E5_REPRESENTATION}_df.pkl'
TF_IDF_REPRESENTATION_PATH = fr'data/{TF_IDF_REPRESENTATION}_dict.pkl'
JACCARD_REPRESENTATION_PATH = fr'data/{JACCARD_REPRESENTATION}_dict.pkl'
BERT_REPRESENTATION_PATH = fr'data/{BERT_REPRESENTATION}_dict.pkl'
COLORS = ['blue', 'orange', 'green']

tqdm.pandas()


def constants_check(df):
    "Check that max rank is MAX_RANK"
    if df['position'].max() != MAX_RANK:
        raise ValueError('Max rank is not {}'.format(MAX_RANK))


def create_combine_df():
    "Read final_df.csv, add user_query column and check constants"
    df = pd.read_csv(r'data/final_df.csv')
    df['user_query'] = df['user'] + '_' + df['query']
    constants_check(df)
    return df


def filter_split_baseline_diversity(df, docno_set=None):
    "Split df into baseline and diversity and remove docno_set if needed"
    assert 'docno' in df.columns and 'user_query' in df.columns and 'user' in df.columns, 'Missing columns in df'
    baseline = df['docno'].str.split('_').str[-2] == '0'
    diversity = df['docno'].str.split('_').str[-2] == '1'  # == ~baseline
    cond = pd.Series(True, index=df.index)
    if docno_set is not None:
        cond = ~df['docno'].isin(docno_set)

    if not cond.sum() < df.shape[0] and docno_set is not None:
        print('No docno were removed')
        print()

    df_baseline = df[baseline & cond]
    df_diversity = df[diversity & cond]
    return df_baseline, df_diversity


def extract_docno(df):
    "Extract docno from representation_df"
    current_columns = df.columns
    col_list = ['round', 'query', 'user', 'user_query', 'position']
    if set(col_list).issubset(current_columns):
        return df
    df.drop(list(set(current_columns).intersection(set(col_list))), axis=1, inplace=True)
    final_df = pd.read_csv(r'data/final_df.csv')
    final_df['user_query'] = final_df['user'] + '_' + final_df['query']
    final_df = final_df[col_list + ['docno']]
    before_len = df.shape[0]
    df = df.merge(final_df, on='docno', how='inner')
    assert before_len == df.shape[0], 'Some rows were lost during merge'
    return df
    

def texts_representation(text, model):
    "Get representation of text using model"
    input_texts = ["query: " + text]  # prefix for use as vector representation 
    embedding = model.encode(input_texts, normalize_embeddings=True)
    return embedding.squeeze()


def create_representation_df(limit_read=None):
    "Create representation df using e5-large-unsupervised model and SentenceTransformer"
    representation_df = pd.read_csv(r'data/final_df.csv')
    if limit_read is not None:
        representation_df = representation_df.iloc[:limit_read]
    model = SentenceTransformer('intfloat/e5-large-unsupervised')
    representation_df['representation'] = representation_df['doc_text'].progress_apply(texts_representation, model)

    representation_df = representation_df[
        ['round', 'query', 'user', 'user_query', 'position', 'docno', 'representation']]
    return representation_df


def load_or_create_representation_df(path=E5_REPRESENTATION_PATH):
    "Load or create representation df if not exists"
    if os.path.exists(path):
        representation_df = pd.read_pickle(filepath_or_buffer=path)
        if type(representation_df) is dict:
            data = [(k, v) for k, v in representation_df.items()]
            representation_df = pd.DataFrame(data, columns=['docno', 'representation'])
            representation_df = extract_docno(representation_df)
    else:
        print("Creating representation df base in e5-large-unsupervised - other models are not supported yet")
        representation_df = create_representation_df()
        representation_df.to_pickle(E5_REPRESENTATION_PATH)

    return representation_df
