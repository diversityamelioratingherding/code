import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os

from utils.general_utils import load_or_create_representation_df, filter_split_baseline_diversity, \
    NUM_ROUNDS, COLORS

NUM_PERMUTATIONS = int(1e5)
MAX_COMPONENTS = 50
MAX_COMPONENTS_GROUPS = 10
ROUNDS_STR = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}


def calc_similarity(vector_i, vector_j, sparse=False, jaccard=False):
    "Calculate the similarity between two vectors"
    similarity = None
    if sparse and not jaccard:
        vector_j = vector_j.T
        similarity = vector_i @ vector_j
        similarity = similarity.toarray().flatten()[0]
    if sparse and jaccard:
        vector_i = csr_matrix(vector_i)
        vector_j = csr_matrix(vector_j)
        similarity_a = vector_i.multiply(vector_j).count_nonzero()
        similarity_b = (vector_i + vector_j).count_nonzero()
        similarity = 1.0 * similarity_a / similarity_b
    if not sparse:
        if jaccard:
            raise ValueError('Jaccard similarity is only supported for non-sparse representations')
        similarity = vector_i @ vector_j
        norm_i = np.linalg.norm(vector_i)
        norm_j = np.linalg.norm(vector_j)
        similarity /= (norm_i * norm_j)
    return similarity


def pairwise_similarity(matrix, representation=None, keys=None):
    "Calculate the pairwise similarity between the rows of a matrix"
    jaccard = False
    path = None
    if representation is not None:
        assert keys is not None, 'Keys should be provided for representation similarity'
        assert len(matrix) == len(keys), 'Matrix and keys should have the same length'
        path = f'data/{representation}_similarity_dict.pkl'
        if os.path.exists(path):
            representation_similarity_dict = pd.read_pickle(path)
        else:
            representation_similarity_dict = dict()
        if 'jaccard' in representation:
            jaccard = True
    else:
        representation_similarity_dict = None
    changed = False

    similarities = []
    sparse = matrix[0].shape[0] == 1
    assert sparse or len(matrix.shape) == 2, 'Matrix should be 2D: ' + str(object=matrix.shape)
    for (i, j) in combinations(range(len(matrix)), 2):
        if i == j:
            continue
        pair = tuple({keys[i], keys[j]})
        if representation_similarity_dict is not None and pair in representation_similarity_dict.keys():
            similarity = representation_similarity_dict[pair]
        else:
            vector_i = matrix[i]
            vector_j = matrix[j]
            similarity = calc_similarity(vector_i, vector_j, sparse, jaccard)
            if representation_similarity_dict is not None:
                representation_similarity_dict[pair] = similarity
                changed = True

        similarities.append(similarity)

    if changed and representation_similarity_dict is not None:
        pd.to_pickle(representation_similarity_dict, path)
    return similarities


def pairwise_similarity_measurements(matrix, functions, representation=None, keys=None):
    "Calculate the pairwise similarity between the rows of a matrix and apply functions to the similarities"
    similarities = pairwise_similarity(matrix, representation, keys)
    return [func(similarities) for func in functions]


def permutation_test(arr1, arr2, num_permutations=NUM_PERMUTATIONS):
    "Calculate the p-value of a permutation test"
    observed_diff = np.abs(arr1.mean() - arr2.mean())  # Calculate the difference between the means of the two lists
    combined = np.concatenate([arr1, arr2])
    arr1_shape = arr1.shape[0]
    count = 0
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        permuted_list1 = combined[:arr1_shape]
        permuted_list2 = combined[arr1_shape:]
        assert len(permuted_list1) == len(arr1) and len(permuted_list2) == len(arr2), 'Permutation error'
        assert len(permuted_list1) > 0 and len(permuted_list2) > 0, 'Permutation error'
        permuted_diff = np.abs((np.mean(permuted_list1) - np.mean(permuted_list2)))
        if permuted_diff >= observed_diff:
            count += 1
    p_value = count / num_permutations
    return p_value


def calc_round_statistics(df, df_name, representation):
    "Calculate the average and minimum similarity between documents in the same group in each round"
    base_groups = df[['query', 'round', 'representation', 'docno']].groupby(['query', 'round'])
    groups = base_groups.filter(lambda x: len(x.values) > 1).groupby(['query', 'round'])

    pairwise_function = lambda x: pairwise_similarity_measurements(np.stack(x['representation'].values),
                                                                   [np.mean, np.min],
                                                                   representation, x['docno'].values)
    similarity_measurements = groups.apply(pairwise_function)
    round_mean_similarities = similarity_measurements.apply(lambda x: x[0])
    round_min_similarities = similarity_measurements.apply(lambda x: x[1])

    res = {'Dataset': df_name,
           'Average round similarity': round_mean_similarities.mean(),
           'Min round similarity': round_min_similarities.mean()}
    return res


def p_value_calc_round_statistics(df_baseline, df_diversity, representation):
    "Calculate the p-value of the average and minimum similarity between documents in the same group in each round"
    base_groups_baseline = df_baseline[['query', 'round', 'representation', 'docno']].groupby(['query', 'round'])
    groups_baseline = base_groups_baseline.filter(lambda x: len(x.values) > 1).groupby(['query', 'round'])
    base_groups_diversity = df_diversity[['query', 'round', 'representation', 'docno']].groupby(['query', 'round'])
    groups_diversity = base_groups_diversity.filter(lambda x: len(x.values) > 1).groupby(['query', 'round'])

    pairwise_function = lambda x: pairwise_similarity_measurements(np.stack(x['representation'].values),
                                                                   [np.mean, np.median, np.max, np.min, np.std],
                                                                   representation, x['docno'].values)
    similarity_measurements_baseline = groups_baseline.apply(pairwise_function)
    round_mean_similarities_baseline = similarity_measurements_baseline.apply(lambda x: x[0])
    round_min_similarities_baseline = similarity_measurements_baseline.apply(lambda x: x[3])

    similarity_measurements_diversity = groups_diversity.apply(pairwise_function)
    round_mean_similarities_diversity = similarity_measurements_diversity.apply(lambda x: x[0])
    round_min_similarities_diversity = similarity_measurements_diversity.apply(lambda x: x[3])

    p_value_mean = permutation_test(np.array(round_mean_similarities_baseline),
                                    np.array(round_mean_similarities_diversity))
    print(f'p_value of round mean similarity: {p_value_mean:.2f}, significant: {p_value_mean < 0.05}')
    p_value_min = permutation_test(np.array(round_min_similarities_baseline),
                                   np.array(round_min_similarities_diversity))
    print(f'p_value of round min similarity: {p_value_min:.2f}, significant: {p_value_min < 0.05}')


def plot_first_second_distance_over_time(df_baseline, df_diversity, representation, max_rounds=NUM_ROUNDS, bars=True):
    "Plot the average distance between the two highest ranked documents over time"
    avg_first_second_similarity = ([], [])
    sd_first_second_similarity = ([], [])
    len_rounds = ([], [])

    dfs = [df_baseline[(df_baseline['position'] == 1) | (df_baseline['position'] == 2)],
           df_diversity[(df_diversity['position'] == 1) | (df_diversity['position'] == 2)]]
    labels = ['Baseline', 'Diversity']
    pairwise_similarity_function = lambda x: np.mean(pairwise_similarity(np.stack(x['representation'].values),
                                                                         representation, x['docno'].values))
    for round_num in range(1, max_rounds + 1):
        for i in range(2):
            df = dfs[i]
            round_df = df[df['round'] == round_num]
            groups = round_df.groupby('query')
            groups = groups.filter(lambda x: len(x.values) > 1).groupby('query')
            round_similarity = groups.apply(pairwise_similarity_function)
            avg_similarity = round_similarity.mean()
            sd_round = round_similarity.std()
            len_round = len(groups)
            avg_first_second_similarity[i].append(avg_similarity)
            sd_first_second_similarity[i].append(sd_round)
            len_rounds[i].append(len_round)

    avg_first_second_similarity = [np.array(object=avg_first_second_similarity[0]), np.array(avg_first_second_similarity[1])]
    sd_first_second_similarity = [np.array(sd_first_second_similarity[0]), np.array(sd_first_second_similarity[1])]
    len_rounds = [np.array(len_rounds[0]), np.array(len_rounds[1])]

    plt.figure(figsize=(6, 6.5))
    for i in range(2):
        plt.plot(range(1, max_rounds + 1), avg_first_second_similarity[i], label=f'{labels[i]}', color=COLORS[i])
        if bars:
            yerr = 1.96 * sd_first_second_similarity[i] / np.sqrt(len_rounds[i])
            plt.errorbar(np.arange(1, max_rounds + 1), avg_first_second_similarity[i], yerr=yerr, fmt='o',
                         color=COLORS[i])

    plt.xlabel('Round', fontsize=14)
    plt.ylabel('Similarity between the two highest ranked documents', fontsize=14)
    if 'jaccard' not in representation:
        plt.ylim((0.4, 1))
    plt.legend(fontsize=14)
    plt.savefig(r'figs\{0}-{1}-{2}-similarity.png'.format(representation, ROUNDS_STR[1],
                                                          ROUNDS_STR[2]), format='png')
    plt.show()
    

def calc_first_second_p_value_double_round(df_baseline, df_diversity, representation, max_rounds=NUM_ROUNDS):
    "Calculate the p-value of the average similarity between the two highest ranked documents in two consecutive rounds"
    dfs = [df_baseline, df_diversity]
    pairwise_similarity_function = lambda x: np.mean(pairwise_similarity(np.stack(x['representation'].values),
                                                                         representation, x['docno'].values))
    for num_round in range(1, max_rounds):
        first_second_group_mean_similarities = []
        ROUNDS = [num_round, num_round + 1]
        for i in range(2):
            df = dfs[i]
            first_second_df = df[df['position'] <= 2]
            first_second_df = first_second_df[first_second_df['round'].isin(ROUNDS)]
            first_second_groups = first_second_df[['query', 'round', 'representation', 'docno']].groupby(
                ['query', 'round'])
            first_second_groups = first_second_groups.filter(lambda x: len(x.values) > 1).groupby(['query', 'round'])
            assert len(first_second_groups.filter(lambda x: len(x.values) > 2)) == 0, 'Error in first-second groups'
            first_second_group_mean_similarities.append(first_second_groups.apply(pairwise_similarity_function))
        p_value_first_second = (permutation_test(np.array(first_second_group_mean_similarities[0]),
                                                 np.array(first_second_group_mean_similarities[1])))
        print(f'Permutation test p-value - average first-second similarity in rounds {num_round, num_round + 1}: '
              f'{p_value_first_second}, significant: {p_value_first_second < 0.05}')


def plot_first_min_similarity_over_time(df_baseline, df_diversity, representation, max_rounds=NUM_ROUNDS):
    "Plot the average similarity of the winner with previous winners over time"
    min_similarity = ([], [])
    dfs = [df_baseline[df_baseline['position'] == 1], df_diversity[df_diversity['position'] == 1]]
    labels = ['Baseline', 'Diversity']
    pairwise_function = lambda x: \
        pairwise_similarity_measurements(np.stack(x['representation'].values), [np.min],
                                         representation, x['docno'].values)
    for round_num in range(2, max_rounds + 1):
        for i in range(2):
            df = dfs[i]
            round_df = df[df['round'] <= round_num]
            groups = round_df.groupby('query')
            groups = groups.filter(lambda x: len(x.values) > 1).groupby('query')
            round_first_measurements = groups.apply(pairwise_function)
            round_min_similarity = round_first_measurements.apply(lambda x: x[0]).mean()
            min_similarity[i].append(round_min_similarity)

    plt.figure(figsize=(6, 6.5))
    for i in range(2):
        plt.plot(range(2, max_rounds + 1), min_similarity[i], label=f'{labels[i]}', color=COLORS[i])
    plt.xlabel('Rounds Included', fontsize=14)
    plt.ylabel('Min similarity of the winner with previous winners', fontsize=14)
    if 'jaccard' not in representation:
        plt.ylim((0.6, 1))
    plt.legend(fontsize=14)
    plt.savefig(r'figs\{0}-agg-min-average-first-similarity.png'.format(representation), format='png')
    plt.show()


def plot_round_pairwise_similarity(df_baseline, df_diversity, representation, with_prev_first=False,
                                   group_position=(1, 2, 3, 4), max_rounds=NUM_ROUNDS):
    "Plot the average similarity between documents in the same group over time and calculate the p-value"
    dfs = [df_baseline, df_diversity]
    labels = ['Baseline', 'Diversity']
    pairwise_function = lambda x: pairwise_similarity_measurements(np.stack(x['representation'].values),
                                                                   [np.mean, np.min],
                                                                   representation, x['docno'].values)
    start_round = 1
    if with_prev_first:
        start_round = 2
    baseline_mean_matrix = []
    diversity_mean_matrix = []
    plt.figure(figsize=(6, 6.5))
    for i in range(2):
        df = dfs[i]
        rounds_avg_similarities = []
        rounds_sd_similarities = []
        len_round = []
        for round_num in range(start_round, max_rounds + 1):
            if with_prev_first:
                round_df = df[((df['round'] == round_num - 1) & (df['position'] == 1)) |
                              ((df['round'] == round_num) & (df['position'].isin(group_position)))]
            else:
                round_df = df[((df['round'] == round_num) & (df['position'].isin(group_position)))]
            groups = round_df.groupby('query')
            groups = groups.filter(lambda x: len(x.values) > 1).groupby('query')
            len_round.append(len(groups))

            round_similarity = groups.apply(pairwise_function)

            round_avg_similarity = round_similarity.apply(lambda x: x[0]).mean()
            round_sd_similarity = round_similarity.apply(lambda x: x[0]).std()

            if i == 0:
                baseline_mean_matrix.append(np.array(round_similarity.apply(lambda x: x[0])))
            elif i == 1:
                diversity_mean_matrix.append(np.array(round_similarity.apply(lambda x: x[0])))
            rounds_avg_similarities.append(round_avg_similarity)
            rounds_sd_similarities.append(round_sd_similarity)

        rounds_avg_similarities = np.array(rounds_avg_similarities)
        plt.plot(range(start_round, max_rounds + 1), rounds_avg_similarities, label=f'{labels[i]}', color=COLORS[i])

    for i, rou in enumerate(range(start_round, max_rounds)):
        p_value_avg = permutation_test(np.concatenate((baseline_mean_matrix[i], baseline_mean_matrix[i + 1])),
                                       np.concatenate((diversity_mean_matrix[i], diversity_mean_matrix[i + 1])))
        print(f'Permutation test p-value - average round pairwise group similarity [aggregate rounds {[rou, rou + 1]}]:'
              f' {p_value_avg}, significant: {p_value_avg < 0.05}')

    plt.xlabel('Round', fontsize=14)
    plt.ylabel('Mean inter-document similarity in a ranked list', fontsize=14)
    if 'jaccard' not in representation:
        plt.ylim([0.35, 1])
    plt.legend(fontsize=14)
    name = 'average-group-similarity'
    if with_prev_first and len(group_position) == 3:
        name = name + "-with-prev-first"
    elif (not with_prev_first) and len(group_position) == 3:
        name = name + "-without-first"
    elif (not with_prev_first) and len(group_position) == 4:
        pass
    else:
        raise Exception("error in group similarity calculation")
    plt.savefig(r'figs\{0}-average-pairwise-{1}.png'.format(representation, name), format='png')
    plt.show()


def plot_consecutive_winners_over_time(df_baseline, df_diversity, representation, max_rounds=NUM_ROUNDS):
    "Plot the similarity between consecutive winners over time and calculate the p-value"
    avg_first_second_similarity = ([], [])
    sd_first_second_similarity = ([], [])
    len_rounds = ([], [])

    dfs = [df_baseline[df_baseline['position'] == 1],
           df_diversity[
               df_diversity['position'] == 1]]
    labels = ['Relevance', 'Diversity']
    pairwise_similarity_function_tag = lambda x: (
        np.mean(pairwise_similarity(np.stack(x['representation'].values), representation, x['docno'].values))) \
        if (len(np.unique(x['user'].values)) > 1) \
        else 2 + np.mean(pairwise_similarity(np.stack(x['representation'].values), representation, x['docno'].values))
    samples_base = 0
    total_samples_base = []
    samples_div = 0
    total_samples_div = []
    for round_num in range(2, NUM_ROUNDS + 1):
        for i in range(2):
            df = dfs[i]
            round_df = df[(df['round'] == round_num) | (df['round'] == round_num - 1)]
            groups = round_df.groupby('query')
            groups = groups.filter(lambda x: len(x.values) > 1).groupby('query')
            round_similarity = groups.apply(pairwise_similarity_function_tag)
            round_similarity = np.array([value for value in round_similarity if value < 2])
            if i == 0:
                total_samples_base = np.concatenate((total_samples_base, round_similarity))
                samples_base = samples_base + len(round_similarity)
            if i == 1:
                total_samples_div = np.concatenate((total_samples_div, round_similarity))
                samples_div = samples_div + len(round_similarity)
            avg_similarity = round_similarity.mean()
            sd_round = round_similarity.std()
            len_round = len(groups)
            avg_first_second_similarity[i].append(avg_similarity)
            sd_first_second_similarity[i].append(sd_round)
            len_rounds[i].append(len_round)

    avg_first_second_similarity = [np.array(avg_first_second_similarity[0]), np.array(avg_first_second_similarity[1])]
    p_val = permutation_test(np.array(total_samples_base), np.array(total_samples_div))
    print(f'permutation_test_consecutive_winners: {p_val:.2f}, significant: {p_val < 0.05}')
    plt.figure(figsize=(6, 6.5))
    for i in range(2):
        plt.plot(range(2, max_rounds + 1), avg_first_second_similarity[i], label=f'{labels[i]}', color=COLORS[i])

    plt.xlabel('Round', fontsize=14)
    plt.ylabel('Similarity between consecutive winners', fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(r'figs\{0}-cons-winners-similarity.png'.format(representation), format='png')
    plt.show()


def content_analysis(path, docno_set=None):
    "Perform content analysis on the given representation"
    np.random.seed(0)
    representation_df = load_or_create_representation_df(path=path)
    df_baseline, df_diversity = \
        filter_split_baseline_diversity(representation_df, docno_set)

    representation = path.split('/')[-1].split('_')[0].replace('.pkl', '')

    names = ['Baseline', 'Diversity']
    dfs = [df_baseline, df_diversity]
    round_results_df = pd.DataFrame()
    for i in range(2):
        res = calc_round_statistics(dfs[i], names[i], representation=representation)
        round_results_df = pd.concat([round_results_df, pd.DataFrame([res])])

    round_results_df.set_index('Dataset', inplace=True)
    round_results_df = round(round_results_df, 2)

    p_value_calc_round_statistics(df_baseline, df_diversity, representation=representation)
    print('\n\n')

    plot_consecutive_winners_over_time(df_baseline, df_diversity, representation)

    plot_first_second_distance_over_time(df_baseline, df_diversity, representation)

    calc_first_second_p_value_double_round(df_baseline, df_diversity, representation)
    print('\n\n')

    plot_round_pairwise_similarity(df_baseline, df_diversity, representation)

    plot_first_min_similarity_over_time(df_baseline, df_diversity, representation)

    return round_results_df
