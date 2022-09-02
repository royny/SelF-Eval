import os
import logging
from typing import List, Dict
from collections import OrderedDict

import numpy as np
import prettytable as pt
from scipy.stats import pearsonr, spearmanr, kendalltau

import copy
from test_data import HumanJudgementDatasetContest


class Evaluator:
    """#TODO: adds docstring
    """

    def __init__(self,
                 checkpoint_dir_path,
                 eval_data_dir_path='./evaluation/eval_data',
                 result_file_name='d_score_tt_results.txt',
                 datasets=['fed-dial', 'dstc9','fed-turn'],
                 eval_mode='mix',
                 score_whole_dialog=True,
                 console_output=True):
        self.checkpoint_dir_path = checkpoint_dir_path
        self.eval_data_dir_path = eval_data_dir_path
        self.datasets = datasets
        self.eval_mode = eval_mode
        self.score_whole_dialog = score_whole_dialog
        self.console_output = console_output
        self.result_file_path = os.path.join(
            checkpoint_dir_path, result_file_name)

        table_header = ['info', 'Pearson', 'Spearman',  'avg']
        self.table_recorder = TableRecorder(table_header, table_names=datasets)

        self.metric_model = None
        self.additional_eval_info = None

    def evaluate(self, metric_model, additional_eval_info=None):
        self.metric_model = metric_model
        self.additional_eval_info = additional_eval_info
        if self.eval_mode == 'mix':
            self._evaluate_mix()
        elif self.eval_mode == 'separate':
            self._evaluate_sep()

    def write_cor_score(self, predicted_scores, human_scores, empty_score_indices, quality_name, dataset_name, diag=False):
        '''
            write correlation score pt 
            if there're empty indices, del them in predicted_scores to compute cor with human_scores.
        '''

        if len(empty_score_indices) == 0 or quality_name not in ['Error recovery', 'error recovery'] :
            correlation_results = self._compute_correlation(predicted_scores, human_scores)
        else:
            pred_scores = copy.copy(predicted_scores)
            for i in reversed(empty_score_indices):
                del pred_scores[i]
            correlation_results = self._compute_correlation(pred_scores, human_scores)

        result_dict = OrderedDict()
        for meta_metric_name, result in correlation_results.items():
            result_dict[meta_metric_name] = '{}({})'.format(
                result[0], result[1])
        result_dict['avg'] = np.average(
            [result[0] for result in correlation_results.values()])
        if diag:
            table_row = [quality_name+'/diag'] + list(result_dict.values())
        else:
            table_row = [quality_name] + list(result_dict.values())
        self.table_recorder.add_row(table_row=table_row,
                                    table_name=dataset_name)

    def _evaluate_mix(self):
        """Uses mix data to evaluate the specified metric model.
        """
        for dataset_name in self.datasets:
            eval_data = HumanJudgementDatasetContest(dataset_name, self.score_whole_dialog)

            if dataset_name in ['persona-see', 'fed-dial', 'dstc9']:
                if self.score_whole_dialog:
                    predicted_scores, human_scores, empty_score_indices, d_score = self._raw_multi_scores(eval_data)
                else:
                    predicted_scores, human_scores, empty_score_indices = self._get_multi_scores(eval_data)
            else:
                predicted_scores, human_scores, d_score = self._get_scores(eval_data)
                empty_score_indices = []

            first_row = [self.additional_eval_info, '', '', '']
            self.table_recorder.add_row(table_row=first_row, table_name=dataset_name)
            if type(human_scores) == list:
                self.write_cor_score(predicted_scores, human_scores, empty_score_indices, 'overall', dataset_name)
                self.write_cor_score(d_score, human_scores, empty_score_indices, 'overall', dataset_name, diag=True)

            else:
                for i in human_scores.keys():
                  self.write_cor_score(predicted_scores, human_scores[i], empty_score_indices, i, dataset_name)
                  self.write_cor_score(d_score, human_scores[i], empty_score_indices, i, dataset_name, diag=True)


            pred_score_dir_path = os.path.join(
                self.checkpoint_dir_path, 'predicted_scores/contest')
            if not os.path.exists(pred_score_dir_path):
                os.makedirs(pred_score_dir_path)

            pred_score_text_name = 'pred_scores_{}_mix_{}.txt'.format(
                dataset_name, self.additional_eval_info)
            pred_score_text_path = os.path.join(
                pred_score_dir_path, pred_score_text_name)
            self._save_txt(predicted_scores, pred_score_text_path)

            pred_score_text_name = 'diag_scores_{}_mix_{}.txt'.format(
                dataset_name, self.additional_eval_info)
            pred_score_text_path = os.path.join(
                pred_score_dir_path, pred_score_text_name)
            self._save_txt(d_score, pred_score_text_path)

            '''
            pred_score_dist_name = 'pred_score_dist_{}_mix_{}.png'.format(
                dataset_name, self.additional_eval_info)
            pred_score_dist_path = os.path.join(
                pred_score_dir_path, pred_score_dist_name)
            self._save_distribution(predicted_scores, pred_score_dist_path)
            '''

        self.table_recorder.save(output_file_path=self.result_file_path)

    def _evaluate_sep(self):
        """Uses separate data to evaluate the specified metric model.
        """
        for dataset_name in self.datasets:
            result_dict = OrderedDict()
            results_of_cur_metric = []
            dialog_model_names = HumanJudgementDataset.DATA_MAP[dataset_name]
            for dialog_model_name in dialog_model_names:
                eval_data = HumanJudgementDataset(
                    self.eval_data_dir_path,
                    dataset_name, dialog_model_name)
                predicted_scores, human_scores = self._get_scores(eval_data)
                correlation_results = self._compute_correlation(
                    predicted_scores, human_scores)

                for meta_metric_name, result in correlation_results.items():
                    key = '{}-{}'.format(
                        dialog_model_name, meta_metric_name)
                    result_dict[key] = '{}({})'.format(result[0], result[1])
                    results_of_cur_metric.append(result[0])

                pred_score_dir_path = os.path.join(
                    self.checkpoint_dir_path, 'predicted_scores')
                if not os.path.exists(pred_score_dir_path):
                    os.makedirs(pred_score_dir_path)

                pred_score_text_name = 'pred_scores_{}_{}_{}.txt'.format(
                    dataset_name, dialog_model_name, self.additional_eval_info)
                pred_score_text_path = os.path.join(
                    pred_score_dir_path, pred_score_text_name)
                self._save_txt(predicted_scores, pred_score_text_path)

                pred_score_dist_name = 'pred_score_dist_{}_{}_{}.png'.format(
                    dataset_name, dialog_model_name, self.additional_eval_info)
                pred_score_dist_path = os.path.join(
                    pred_score_dir_path, pred_score_dist_name)
                self._save_distribution(predicted_scores, pred_score_dist_path)

            result_dict['avg'] = np.average(results_of_cur_metric)
            table_row = [self.additional_eval_info] + list(result_dict.values())
            self.table_recorder.add_row(table_row=table_row,
                                        table_name=dataset_name)

        self.table_recorder.save(output_file_path=self.result_file_path)

    @staticmethod
    def _compute_correlation(predicted_scores: List[float],
                             human_scores: List[float]) -> Dict[str, str]:
        pearson_r, pearson_p = pearsonr(predicted_scores, human_scores)
        spearman_r, spearman_p = spearmanr(predicted_scores, human_scores)
        #kendall_tau, kendall_p = kendalltau(predicted_scores, human_scores)
        correlation_results = {
            'Pearson': (round(pearson_r, 3), round(pearson_p, 3)),
            'Spearman': (round(spearman_r, 3), round(spearman_p, 3)),
            #'Kendall': (round(kendall_tau, 3), round(kendall_p, 3)),
        }
        return correlation_results

    def _raw_multi_scores(self, eval_data):
        '''
            Get predicted score, human golden score and empty index which is because of the lacking 'Error Recovery' quality in FED.

        '''

        predicted_scores = []
        multi_human_scores = {}
        empty_score_indices = []
        d_score = []

        for num, sample in enumerate(eval_data):
            dialog = '</s></s>'.join(sample['dialog'])
            agg_score, dscore = self.metric_model.get_score(dialog)
            agg_score = round(agg_score,2)
            dscore = round(dscore,2)
            predicted_scores.append(agg_score)
            d_score.append(dscore)

            for qualitiy, score in sample['human_score'].items():
                if qualitiy not in multi_human_scores:
                    multi_human_scores[qualitiy] = []
                if type(score)==float:  
                    multi_human_scores[qualitiy].append(score)
                else:
                    empty_score_indices.append(num)

        return predicted_scores, multi_human_scores, empty_score_indices, d_score

    def _get_multi_scores(self, eval_data, mode='avg'):
        '''
            adapt for single-turn evaluation model, aggerate turn-level score to a dialog-level score.
            'mode' here denotes aggagregation strategy
        '''
        predicted_scores = []
        multi_human_scores = {}
        empty_score_indices = []

        for num, sample in enumerate(eval_data):
            predicted_score = [self.metric_model.get_score(i) for i in sample['turns']]
            agg_score = round(sum(predicted_score)/len(predicted_score), 2)
            predicted_scores.append(agg_score)

            for qualitiy, score in sample['human_score'].items():
                    if qualitiy not in multi_human_scores:
                        multi_human_scores[qualitiy] = []
                    if type(score)==float:  
                        multi_human_scores[qualitiy].append(score)
                    else:
                        empty_score_indices.append(num)

        return predicted_scores, multi_human_scores, empty_score_indices

    def _get_scores(self, eval_data):
        predicted_scores = []
        human_scores = []
        multi_human_scores = {}
        
        d_score = []

        flag = 0
        for sample in eval_data:
            dialog = '</s></s>'.join(sample['context'])
            dialog = dialog+'</s></s>'+sample['hyp_response']
            predicted_score, dscore = self.metric_model.get_score(dialog)
            predicted_scores.append(predicted_score)

            d_score.append(dscore)

            if type(sample['human_score'])==float:
                human_score = sample['human_score']
                human_scores.append(human_score)
            else:
                flag = 1
                for qualitiy, score in sample['human_score'].items():
                    if qualitiy not in multi_human_scores:
                        multi_human_scores[qualitiy] = []
                    multi_human_scores[qualitiy].append(score)
                    
        if flag == 0:
            return predicted_scores, human_scores, d_score
        else:
            return predicted_scores, multi_human_scores, d_score

    def _save_txt(self, content: list, output_file_path):
        if self.console_output:
            print('Saving text into {}...'.format(output_file_path))
        with open(output_file_path, 'w') as f:
            for element in content:
                f.write(str(element) + '\n')

    def _save_distribution(self, content: List[float], output_file_path):
        if self.console_output:
            print('Saving distribution into {}...'.format(output_file_path))
        content = [round(element, 8) for element in content]
        sns.set_style('darkgrid')
        plt.figure()
        ax = sns.histplot(content, kde=True)
        ax.set_xlabel('score')
        ax.set_ylabel('count')
        plt.savefig(output_file_path)


class TableRecorder:

    def __init__(self, table_header: list, table_names: List[str]):
        self.table_dict = {
            name: pt.PrettyTable(table_header) for name in table_names
        }

    def add_row(self, table_row: list, table_name: str):
        self.table_dict[table_name].add_row(table_row)

    def save(self, output_file_path: str):
        with open(output_file_path, 'w') as f:
            for table_name, table in self.table_dict.items():
                f.write(table_name + '\n')
                f.write(str(table) + '\n\n')
