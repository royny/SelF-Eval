import json
import functools
from typing import List, Dict
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import texar.torch as tx

from trainer import Trainer


class mlr_Trainer(Trainer):

    def __init__(self, model, dataset, val_dataset, args):
        super().__init__(model, dataset, val_dataset, args)
        #nn and loss parameters
        self.warmup_proportion = args.warmup_proportion
        self.centroid_mode = args.centroid_mode
        self.feature_distance_mode = args.distance_mode
        self.feature_distance_lower_bound = args.feature_distance_lower_bound
        self.feature_distance_upper_bound = args.feature_distance_upper_bound
        self.score_distance_lower_bound = eval(args.score_distance_lower_bound)
        self.score_distance_upper_bound = args.score_distance_upper_bound
        self.weighted_s_loss = args.weighted_s_loss
        self.feature_loss_weight = args.feature_loss_weight
        self.score_loss_weight = args.score_loss_weight
        self.bce_loss_weight = args.bce_loss_weight
        self.use_projection_head = args.use_projection_head

        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        self.cur_distance_mode = None
        self.cur_upper_bound = None
        self.cur_lower_bound = None

        self.bce_criterion = torch.nn.BCELoss()

        self.device = torch.device("cuda".format(args.gpu))

        if self.use_projection_head:
            hidden_size = self.model.get_hidden_size()
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size))
            self.projection_head.to(self.device)

    
    def _get_optimizer(self):
        vars_with_decay = []
        vars_without_decay = []
        for name, param in self.model.named_parameters():
            if 'layer_norm' in name or name.endswith('bias'):
                vars_without_decay.append(param)
            else:
                vars_with_decay.append(param)

        opt_params = [{
            'params': vars_with_decay,
            'weight_decay': 0.01,
        }, {
            'params': vars_without_decay,
            'weight_decay': 0.0,
        }]

        optimizer = tx.core.BertAdam(
            opt_params,
            betas=(0.9, 0.999),
            eps=1e-6,
            lr=self.learning_rate
        )
        return optimizer

    def _get_scheduler(self):
        def get_lr_multiplier(step: int, total_steps: int, warmup_steps: int):
            """Calculate the learning rate multiplier given current step and the
            number of warm-up steps. The learning rate schedule follows a linear
            warm-up and linear decay.
            """
            step = min(step, total_steps)

            multiplier = (1 - (step-warmup_steps) / (total_steps-warmup_steps))

            if warmup_steps > 0 and step < warmup_steps:
                warmup_percent_done = step / warmup_steps
                multiplier = warmup_percent_done

            return multiplier

        lr_func = functools.partial(get_lr_multiplier,
                                    total_steps=self.num_total_train_steps,
                                    warmup_steps=self.num_warmup_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_func)
        return scheduler


    def _train_epoch(self):
        self.model.train()
        avg_recorder = tx.utils.AverageRecorder(size=self.display_steps)
        for batch, sep in tqdm(self.dataset, total=len(self.dataset)):
            batch_size = self.get_batch_size(batch)
            all_features, all_scores, all_ds = self._get_all_features_and_scores(batch, sep)
            
            #loss, loss_info_dict = self._compute_dual_mlr_loss(
             #   all_features, all_scores)
            dloss, dloss_info_dict = self._compute_dual_mlr_loss(
                all_features, all_ds)

            #aloss = loss+dloss
            avg_recorder.add(dloss_info_dict, batch_size)
            dloss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            # updates logging information
            step = self.global_step[Trainer.TRAIN]
            '''
            if self._is_time_to_display(step):
                avg_loss_info_dict = avg_recorder.avg()
                lr_info_dict = {'lr': self.cur_lr}
                train_info_dict = dict(avg_loss_info_dict, **lr_info_dict)
                train_info = self._get_train_info(train_info_dict)
                self.logger.info(train_info)
            '''
            for loss_name, loss_value in dloss_info_dict.items():
                self.tensorboard_writer.add_scalar(
                    '{}/train'.format(loss_name), loss_value, step)
            self.global_step[Trainer.TRAIN] += 1
            train_results = self._get_metric_results()
        return train_results

    @torch.no_grad()
    def _eval_epoch(self, mode):
        self.model.eval()
        avg_recorder = tx.utils.AverageRecorder()
        '''
        self.feature_visualizer.reset()
        self.score_visualizer.reset()
        '''
        for batch in tqdm(self.val_dataset, total=len(self.val_dataset)):
            batch_size = self.get_batch_size(batch)
            all_features, all_scores = self._get_all_features_and_scores(batch)
            _, loss_info_dict = self._compute_dual_mlr_loss(
                all_features, all_scores)
            avg_recorder.add(loss_info_dict, batch_size)
            step = self.global_step[mode]
            self.global_step[mode] += 1
            if mode == Trainer.VALIDATION:
                for loss_name, loss_value in loss_info_dict.items():
                    self.tensorboard_writer.add_scalar(
                        '{}/valid'.format(loss_name), loss_value, step)
        global_step = 'epoch{}'.format(self.cur_epoch_id)
        '''
        self.feature_visualizer.write(
            global_step=global_step, tag=mode)
        self.score_visualizer.write(
            global_step=global_step, tag=mode)
        '''
        eval_results = self._get_metric_results(avg_recorder)
        self.tensorboard_writer.add_scalar(
                        'full/valid', list(eval_results.values())[0], self.cur_epoch_id)
        return eval_results


    def _get_all_features_and_scores(
        self,
        batch: Dict[str, torch.Tensor], sep) -> List[torch.Tensor]:
        """#TODO
        """
        all_features = []
        all_scores = []
        all_ds = []
        cluster_num = batch['attention_mask'].size()[1]
        
        #batch = {k: v.permute(1,0,2,3).reshape(1,3,-1,512) for k,v in batch.items()}
        for cluster_name in range(cluster_num):
            
            outputs, ds = self._get_features_and_scores_in_specific_cluster(
                batch, cluster_name, sep)
            scores_in_cur_cluster = outputs
            
            #all_features.append(features_in_cur_cluster)
            all_scores.append(scores_in_cur_cluster)
            all_ds.append(ds)
            '''
            self.logger.debug('features_in_cur_cluster({}) size: {}'.format(
                    cluster_name, features_in_cur_cluster.size()))
            self.logger.debug('scores_in_cur_cluster({}) size: {}'.format(
                    cluster_name, scores_in_cur_cluster.size()))
            self.feature_visualizer.add(
                features_in_cur_cluster, self.LEVEL_ID_MAP[cluster_name])
            self.score_visualizer.add(
                scores_in_cur_cluster, self.LEVEL_ID_MAP[cluster_name])
            '''
        
        return all_features, all_scores, all_ds

    def _get_features_and_scores_in_specific_cluster(
        self,
        batch: Dict[str, torch.Tensor],
        cluster_name: str, sep) -> torch.Tensor: 
        """
        now cluster_name is int, which represents the level of cluster, lower the better.
        """

        '''
        input_ids_in_cur_cluster = [
            batch[key]
            for key in batch.keys()
            if cluster_name + '_input_ids' in key
        ]
        token_type_ids_in_cur_cluster = [
            batch[key]
            for key in batch.keys()
            if cluster_name + '_token_type_ids' in key
        ]
        attention_mask_in_cur_cluster = [
            batch[key]
            for key in batch.keys()
            if cluster_name + '_attention_mask' in key
        ]
        '''
        input_ids_in_cur_cluster = batch['input_ids'][0][cluster_name].to(self.device)
        attention_mask_in_cur_cluster = batch['attention_mask'][0][cluster_name].to(self.device)
        t_sep = sep[0][cluster_name].to(self.device)
        
        # _, score_list = self.model(input_ids_in_cur_cluster, attention_mask_in_cur_cluster)
        #feature_list = []
        score_list = []
        ds_list = []
        for input_ids, attention_mask, indice in zip(
            input_ids_in_cur_cluster,
            attention_mask_in_cur_cluster, t_sep):
            
            output_dict, score, diag_score = self.model(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), indice)
            '''
            feature = output_dict['pooler_output']
            if self.use_projection_head:
                feature = self.projection_head(feature)
            feature = feature.unsqueeze(0)
            score = score.unsqueeze(0)
            feature_list.append(feature)
            '''
            score_list.append(score)
            ds_list.append(diag_score)
            '''
            self.logger.debug('feature size: {}'.format(feature.size()))
            self.logger.debug('score size: {}'.format(score.size()))
            '''
        
        #print(score_list)
        #features_in_cur_cluster = torch.cat(feature_list, dim=0)
        scores_in_cur_cluster = torch.cat(score_list, dim=0)
        ds_cur = torch.cat(ds_list, dim=0)
        '''
        print(scores_in_cur_cluster)
        sys.exit()
        '''
        return scores_in_cur_cluster, ds_cur #features_in_cur_cluster,score_list

    def _compute_dual_mlr_loss(self,
                               all_features: List[torch.Tensor],
                               all_scores: List[torch.Tensor]) -> torch.Tensor:
        """#TODO
        all_features and all_scores must be sorted, inside which the former one
        is better than the latter one.
        """
        '''
        self.cur_distance_mode = self.feature_distance_mode
        self.cur_lower_bound = self.feature_distance_lower_bound
        self.cur_upper_bound = self.feature_distance_upper_bound
        feature_s_loss = self._compute_separation_loss(all_features)
        feature_c_loss = self._compute_compactness_loss(all_features)
        feature_o_loss = self._compute_order_loss(all_features)
        feature_s_loss *= self.feature_loss_weight
        feature_c_loss *= self.feature_loss_weight
        feature_o_loss *= self.feature_loss_weight
        feature_mlr_loss = feature_s_loss + feature_c_loss + feature_o_loss
        
        if len(all_scores) == 2:
            score_mlr_loss = self._compute_bce_loss(all_scores)
            print(score_mlr_loss.item())
            sys.exit()
            loss_info_dict = {
                'bce_loss': score_mlr_loss.item(),
                'dual_mlr_loss': score_mlr_loss.item(),
                'score_mlr_loss': 0,
                'score_s_loss': 0,
                'score_c_loss':0,
                'score_o_loss': 0
            }
            
            return score_mlr_loss, loss_info_dict
        '''
        self.cur_distance_mode = 'l1'
        self.cur_lower_bound = self.score_distance_lower_bound
        self.cur_upper_bound = self.score_distance_upper_bound
        score_s_loss = self._compute_separation_loss(all_scores)
        score_c_loss = self._compute_compactness_loss(all_scores)
        score_o_loss = self._compute_order_loss(all_scores)
        score_s_loss *= self.score_loss_weight
        score_c_loss *= self.score_loss_weight
        score_o_loss *= self.score_loss_weight
        score_mlr_loss = score_s_loss + score_c_loss + score_o_loss
        
        #bce_loss = self._compute_bce_loss(all_scores)
        #bce_loss *= self.bce_loss_weight
        
        
        dual_mlr_loss = score_mlr_loss #+ bce_loss + feature_mlr_loss
        loss_info_dict = {
            'dual_mlr_loss': dual_mlr_loss.item(),
            #'bce_loss': bce_loss.item(),
            #'feature_mlr_loss': feature_mlr_loss.item(),
            'score_mlr_loss': score_mlr_loss.item(),
            #'feature_s_loss': feature_s_loss.item(),
            'score_s_loss': score_s_loss.item(),
            #'feature_c_loss': feature_c_loss.item(),
            'score_c_loss': score_c_loss.item(),
            #'feature_o_loss': feature_o_loss.item(),
            'score_o_loss': score_o_loss.item(),
        }
        return dual_mlr_loss, loss_info_dict

    def _compute_bce_loss(self, all_scores):
        positive_scores = all_scores[0]
        positive_scores = positive_scores.view(-1, positive_scores.size(-1))
        negative_scores = torch.cat(all_scores[1:])   #all_scores[-1]
        negative_scores = negative_scores.view(-1, negative_scores.size(-1))
        one_labels = torch.ones_like(positive_scores)
        zero_labels = torch.zeros_like(negative_scores)
        scores = torch.cat([positive_scores, negative_scores])
        labels = torch.cat([one_labels, zero_labels])
        bce_loss = self.bce_criterion(scores, labels)
        return bce_loss


    def _compute_separation_loss(self, all_data_points):
        inter_distances, inter_weights = self._compute_inter_cluster_distances(
            all_data_points)
        if inter_distances.size()[0] == 1:
            inter_bounds = self.cur_lower_bound[0]
        elif inter_distances.size()[0] == 3:
            inter_bounds = self.cur_lower_bound[1]
        else:
            inter_bounds = self.cur_lower_bound[2]

        if self.weighted_s_loss:
            inter_bounds *= inter_weights
        separation_loss = F.relu(inter_bounds - inter_distances)
        separation_loss = separation_loss.sum(dim=0).mean()
        return separation_loss

    def _compute_compactness_loss(self, all_data_points):
        intra_distances = self._compute_intra_cluster_distances(all_data_points)
        compactness_loss = F.relu(
            intra_distances - self.cur_upper_bound)
        compactness_loss = compactness_loss.sum(dim=0).mean()
        return compactness_loss

    def _compute_order_loss(self, all_data_points):
        centroids = []
        for data_points_in_cur_cluster in all_data_points:
            centroid = self.centroid_function(data_points_in_cur_cluster)
            centroids.append(centroid)
        order_loss = None
        for i, better_centroid in enumerate(centroids):
            for worse_centroid in centroids[i+1:]:
                cur_order_loss = F.relu(
                    worse_centroid.norm(dim=-1) - better_centroid.norm(dim=-1))
                if order_loss is None:
                    order_loss = cur_order_loss
                else:
                    order_loss += cur_order_loss
        order_loss = order_loss.mean()
        return order_loss

    def _compute_inter_cluster_distances(self,
                                         all_data_points) -> torch.Tensor:
        inter_distance_list = []
        inter_weight_list = []
        centroids = []
        for data_points_in_cur_cluster in all_data_points:
            centroid = self.centroid_function(data_points_in_cur_cluster)
            centroids.append(centroid)
            #self.logger.debug('centroid size: {}'.format(centroid.size()))
        for i, better_centroid in enumerate(centroids):
            for j, worse_centroid in enumerate(centroids[i+1:]):
                inter_distance = self.distance_function(
                    better_centroid, worse_centroid)
                inter_distance = inter_distance.unsqueeze(0)
                inter_weight = torch.empty_like(inter_distance)
                inter_weight[0] = j + 1
                inter_distance_list.append(inter_distance)
                inter_weight_list.append(inter_weight)
                '''
                self.logger.debug(
                    'inter_distance size: {}'.format(inter_distance.size()))
                self.logger.debug(
                    'inter_distance: {}'.format(inter_distance))
                self.logger.debug(
                    'inter_weight size: {}'.format(inter_weight.size()))
                self.logger.debug(
                    'inter_weight: {}'.format(inter_weight))
                '''
        inter_distances = torch.cat(inter_distance_list, dim=0)
        inter_weights = torch.cat(inter_weight_list, dim=0)
        '''
        self.logger.debug('inter_distances size: {}'.format(
            inter_distances.size()))
        self.logger.debug('inter_distances: {}'.format(
            inter_distances))
        self.logger.debug('inter_weights size: {}'.format(
            inter_weights.size()))
        self.logger.debug('inter_weights: {}'.format(
            inter_weights))
        '''
        return inter_distances, inter_weights

    def _compute_intra_cluster_distances(self,
                                         all_data_points) -> torch.Tensor:
        intra_distance_list = []
        for data_points_in_cur_cluster in all_data_points:
            centroid = self.centroid_function(data_points_in_cur_cluster)
            num_data_points = data_points_in_cur_cluster.size(0)
            repeated_centroids = centroid.repeat(
                num_data_points, 1, 1)
            distances = self.distance_function(
                data_points_in_cur_cluster, repeated_centroids)
            intra_distance = distances.mean(dim=0)
            intra_distance = intra_distance.unsqueeze(0)
            intra_distance_list.append(intra_distance)
            '''
            self.logger.debug(
                'repeated_centroids size: {}'.format(
                    repeated_centroids.size()))
            self.logger.debug(
                'distances size: {}'.format(distances.size()))
            self.logger.debug(
                'intra_distance size: {}'.format(intra_distance.size()))
            '''
        intra_distances = torch.cat(intra_distance_list, dim=0)
        '''
        self.logger.debug('intra_distances size: {}'.format(
            intra_distances.size()))
        self.logger.debug('intra_distances: {}'.format(
            intra_distances))
        '''
        return intra_distances

    def _get_metric_results(self, avg_recorder=None):
        """#TODO
        """
        if avg_recorder:
            avg_dual_mlr_loss = avg_recorder.avg('dual_mlr_loss')
            metric_results = {
                'dual_mlr_loss': avg_dual_mlr_loss,
            }
        else:
            metric_results = {}
        return metric_results


    @property
    def num_warmup_steps(self):
        return self.num_train_steps_per_epoch * self.warmup_proportion

    @property
    def centroid_function(self):
        func_name = 'get_centroid_{}'.format(self.centroid_mode)
        return getattr(self, func_name)

    @property
    def distance_function(self):
        func_name = 'get_distance_{}'.format(self.cur_distance_mode)
        return getattr(self, func_name)

    @staticmethod
    def get_batch_size(batch: Dict[str, torch.Tensor]):
        tensor = list(batch.values())[0]
        return tensor.size(0)

    @staticmethod
    def get_centroid_mean(features):
        
        centroid_feature = features.mean(dim=0)
        return centroid_feature

    @staticmethod
    def get_distance_cosine(vector1, vector2):
        cosine_distance = 1 - F.cosine_similarity(vector1, vector2, dim=-1)
        return cosine_distance

    @staticmethod
    def get_distance_l1(vector1, vector2):
        l1_distance = torch.abs(vector1 - vector2).squeeze(-1)
        return l1_distance