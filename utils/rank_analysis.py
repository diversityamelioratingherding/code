import numpy as np
from utils.general_utils import create_combine_df, filter_split_baseline_diversity, NUM_ROUNDS, MAX_RANK


def create_baseline_diversity_df_rank():
    "Create baseline and diversity dataframes for rank analysis"
    df = create_combine_df()
    df_baseline, df_diversity = filter_split_baseline_diversity(df)

    df_baseline = df_baseline[['user_query', 'round', 'position']]
    df_diversity = df_diversity[['user_query', 'round', 'position']]
    
    assert df_baseline.shape[0] == len(df_baseline['user_query'].unique()) * NUM_ROUNDS
    assert df_diversity.shape[0] == len(df_diversity['user_query'].unique()) * NUM_ROUNDS
    
    df_baseline = df_baseline.pivot(index='user_query', columns='round', values='position')
    df_baseline.reset_index(inplace=True)
    df_baseline.index.name = 'index'
    df_diversity = df_diversity.pivot(index='user_query', columns='round', values='position')
    df_diversity.reset_index(inplace=True)
    df_diversity.index.name = 'index'
    return df_baseline, df_diversity


def calc_prob(df, curr, diff=1):
    "Calculate probability of position curr in round i given position curr in round i-diff"
    prob = [0] * MAX_RANK
    for i in range(1, NUM_ROUNDS + 1 - diff):
        times = (df[i] == curr)
        for pos in range(1, MAX_RANK + 1):
            prob[pos - 1] += (times & (df[i + diff] == pos)).sum()
    prob = np.array(prob) / sum(prob)
    return prob


def prob_analysis(df, text=None):
    "Print probability, expected value and standard deviation for each position transition"
    if text is not None:
        print(text)
    for i in range(1, MAX_RANK + 1):
        prob = calc_prob(df, i)
        expected_value = (prob * np.arange(1, MAX_RANK + 1)).sum()
        variance = (prob * (np.arange(1, MAX_RANK + 1) ** 2)).sum() - expected_value ** 2
        sd = np.sqrt(variance)
        print(f'Position {i}: {prob.round(2)}, expected value: {expected_value:.2f}, sd: {sd:.2f}')


def rank_analysis():
    "Run rank analysis for the baseline and diversity datasets"
    df_baseline, df_diversity = create_baseline_diversity_df_rank()
    prob_analysis(df_baseline, 'Baseline:')
    print()
    prob_analysis(df_diversity, 'Diversity:')
