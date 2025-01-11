FEATURES_ORDER = ['docno', 'qid', 'round', 'user', 'position', 'contents_analyzed_BM25Stat_sum_k1_0.90_b_0.40',
                             'contents_analyzed_LmDirStat_sum_mu_1000', 'contents_analyzed_TfStat_sum',
                             'contents_analyzed_NormalizedTfStat_sum', 'contents_DocSize',
                             'frac_stop', 'stop_cover', 'entropy', 'bert_score', 'add-pw-query', 'add-pw-stop_word',
                             'add-pw-not', 'add-npw-query', 'add-npw-stop_word', 'add-npw-not', 'rmv-pw-query',
                             'rmv-pw-stop_word', 'rmv-pw-not', 'rmv-npw-query', 'rmv-npw-stop_word', 'rmv-npw-not']

COLUMNS_TO_FEATURE_NAMES = {
        'docno': 'document_id',
        'qid': 'query_id',
        'contents_analyzed_BM25Stat_sum_k1_0.90_b_0.40': 'Okapi',
        'contents_analyzed_LmDirStat_sum_mu_1000': 'LM',
        'contents_analyzed_TfStat_sum': 'TF',
        'contents_analyzed_NormalizedTfStat_sum': 'NormTF',
        'contents_DocSize': 'LEN',
        'frac_stop': 'FracStop',
        'stop_cover': 'StopCover',
        'entropy': 'ENT',
        'bert_score': 'BERT'
    }

FEATURES_TO_NORMALIZE = ['Okapi', 'LM', 'TF', 'NormTF', 'LEN', 'FracStop', 'StopCover', 'ENT', 'BERT']
