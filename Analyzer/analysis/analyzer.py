import os
import logging

import pandas as pd
import matplotlib.pyplot as plt

from .data_cleaner import DataCleaner
from .feature_history import FeatureHistoryRetriever
from .graphing import Graphing


class Analyzer:
    """
    Class responsible for orchestrating the analysis of features and generating graphs.
    """

    def __init__(self, features_df_path: str, experiment_folder: str):
        """
        Initialize the Analyzer with the necessary data and prepare the environment.

        :param features_df_path: Path to the CSV file containing feature data.
        :param game_history_df_path: Path to the CSV file containing game history data.
        :param experiment_folder: Path to the folder where results will be saved.
        """

        # Load the CSV files into DataFrames
        features_df = pd.read_csv(features_df_path)
        # game_history_df = pd.read_csv(game_history_df_path)

        # Clean and merge the data
        cleaner = DataCleaner()
        self.merge_df = features_df.rename(columns={"document_id":'docno', "user": 'player', "query_id": 'qid'})
        self.merge_df['is_winner'] = self.merge_df.groupby(['qid', 'round'])['position'].transform(lambda x: x == x.min())
        self.merge_df['subtopic'] = self.merge_df['docno'].apply(lambda x: x.split('_')[-2][0])
        self.merge_df['qid'] = self.merge_df['qid'].apply(lambda x: str(x).zfill(3))
        self.merge_df['query_id_with_subtopic'] = self.merge_df.apply(lambda x: x['qid'] + '_' + x['subtopic'], axis=1)
        # self.merge_df['qid'] = self.merge_df['query_id_with_subtopic']
        # QUERIES_B = ['124', '144', '161', '164', '166', '170', '194', '055', '061', '064', '074', '075', '083', '096', '021']
        QUERIES_A = ['034', '048', '069', '167', '180', '182', '193', '195', '009', '017', '029', '045', '059', '078', '098']
        # QUERIES_ALL = QUERIES_A + QUERIES_B
        queries = 'A'
        # subtopic = 'baseline'
        subtopic = 'diversity'
        self.merge_df = self.merge_df[(self.merge_df['subtopic'] == '1') & (self.merge_df['qid'].isin(QUERIES_A))]
        # Initialize the graphing utility
        self.graphing = Graphing(experiment_folder, queries, subtopic)

        # Set up the experiment folder and subfolders
        self.experiment_folder = os.path.join(experiment_folder, "graphs")
        os.makedirs(self.experiment_folder, exist_ok=True)
        os.makedirs(f'{self.experiment_folder}/query_independent', exist_ok=True)
        os.makedirs(f'{self.experiment_folder}/query_dependent', exist_ok=True)
        logging.info("Experiment folder and subfolders set up.")

    def analyze(self, num_rounds_back: int) -> None:
        """
        Perform the analysis, including generating general graphs and strategy-specific graphs.

        :param num_rounds_back: Number of rounds back to consider for strategy analysis.
        """
        logging.info("Starting analysis...")
        self.__plot_general_graphs()
        self.__plot_strategies(num_rounds_back)
        logging.info("Analysis completed.")

    def __plot_general_graphs(self) -> None:
        """
        Generate and save general analysis graphs, including match wins and consecutive wins.
        """
        fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [5, 3]})
        self.graphing.plot_number_matches_won(self.merge_df, ax=ax[0])
        self.graphing.plot_consecutive_matches_won(self.merge_df, ax=ax[1])
        fig.tight_layout()
        fig.savefig(f'{self.experiment_folder}/number_non_and_consecutive_matches_won.png')
        logging.info("General graphs generated and saved.")

    def __plot_strategies(self, num_rounds_back: int) -> None:
        """
        Plot strategy-specific graphs by analyzing feature trends over time.

        :param num_rounds_back: Number of rounds back to consider for strategy analysis.
        """
        self.__plot_query_independent_features(num_rounds_back)
        self.__plot_query_dependent_features(num_rounds_back)
        logging.info("Strategy-specific graphs generated.")

    def __plot_query_independent_features(self, num_rounds_back: int) -> None:
        """
        Plot query-independent feature analysis and generate LaTeX figures.

        :param num_rounds_back: Number of rounds back to consider for analysis.
        """
        query_independent_features = {
            'ENT': 'Entropy',
            'FracStop': 'StopWordsRatio',
            'StopCover': 'StopWordsCover',
            'LEN': 'DocumentLength',
        }

        fig_caption = "Comparison of averaged query-independent feature values for documents that shifted from losing to winning..."
        latex_figure = """\\begin{figure}\n\\centering\n"""

        FeatureHistoryRetriever.query_dependent = False

        with_legend = True
        for feature in query_independent_features:
            feature_title = query_independent_features[feature]
            feature_filename = f'{self.experiment_folder}/query_independent/{feature_title.split(".")[-1].lower()}_improvers.jpg'
            self.graphing.plot_feature_improvers(
                self.merge_df, feature, num_rounds_back=num_rounds_back, save_path=feature_filename,
                with_legend=with_legend
            )

            latex_figure += f'\\begin{{subfigure}}{{\\linewidth}}\n  \\centering\n  \\includegraphics[width=\\linewidth]{{../figures/{feature_filename.split(".")[0]}}}\n  \\caption{{{feature_title}}}\n  \\label{{fig:{feature_title.lower()}}}\n\\end{{subfigure}}%\n\\medskip\n\n'

        latex_figure += f'\\caption{{Query Independent Features: {fig_caption}}}\n\\label{{fig:query-independent}}\n\\end{{figure}}'
        with open(f'{self.experiment_folder}/query_independent/query_independent.tex', 'w') as f:
            f.write(latex_figure)
        logging.info("Query-independent features plotted and LaTeX figure generated.")

    def __plot_query_dependent_features(self, num_rounds_back: int) -> None:
        """
        Plot query-dependent feature analysis and generate LaTeX figures.

        :param num_rounds_back: Number of rounds back to consider for analysis.
        """
        query_dependent_features = {
            'Okapi': 'Okapi',
            'LM': 'LanguageModel',
            'TF': 'TermFrequency',
            'NormTF': 'NormalizedTermFrequency',
            'BERT': 'BERT'
        }

        fig_caption = "Comparison of averaged query-dependent feature values for documents that shifted from losing to winning..."
        latex_figure = """\\begin{figure}\n\\centering\n"""

        FeatureHistoryRetriever.query_dependent = True

        with_legend = True
        for feature in query_dependent_features:
            feature_title = query_dependent_features[feature]
            feature_filename = f'{self.experiment_folder}/query_dependent/{feature_title.split(".")[-1].lower()}_improvers.jpg'
            self.graphing.plot_feature_improvers(
                self.merge_df, feature, num_rounds_back=num_rounds_back, save_path=feature_filename,
                with_legend=with_legend
            )

            latex_figure += f'\\begin{{subfigure}}{{\\linewidth}}\n  \\centering\n  \\includegraphics[width=\\linewidth]{{../figures/{feature_filename.split(".")[0]}}}\n  \\caption{{{feature_title}}}\n  \\label{{fig:{feature_title.lower()}}}\n\\end{{subfigure}}%\n\\medskip\n\n'

        latex_figure += f'\\caption{{Query Dependent Features: {fig_caption}}}\n\\label{{fig:query-dependent}}\n\\end{{figure}}'
        with open(f'{self.experiment_folder}/query_dependent/query_dependent.tex', 'w') as f:
            f.write(latex_figure)
        logging.info("Query-dependent features plotted and LaTeX figure generated.")

    def __build_df(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge and process the feature and game history DataFrames.

        :param features_df: Cleaned DataFrame of features.
        :param game_history_df: Cleaned DataFrame of game history.
        :return: Merged and processed DataFrame.
        """
        merged_df = pd.merge(
            features_df, game_history_df,
            left_on=['query_id', 'round', 'user'],
            right_on=['query_id', 'round', 'player'],
            how='left'
        )

        merged_df.rename(columns={'rank': 'position', 'document_id': 'docno'}, inplace=True)
        merged_df['is_winner'] = merged_df.groupby(['qid', 'round'])['position'].transform(lambda x: x == x.min())

        merged_df.drop(['query_id', 'user', 'document', 'prompt', 'points', 'game_id'], axis=1, inplace=True)

        return merged_df
