import os
import random
import logging
import shutil
import sys

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Union

from evaluator import Evaluator
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


class Trainer:
    """#TODO: adds docstring
    """

    TRAIN = 'train'
    VALIDATION = 'valid'
    TEST = 'test'

    def __init__(self, model, dataset, val_dataset, args):
        # constants
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.gpu = args.gpu
        if os.path.isdir(args.checkpoint_dir_path):
            self.checkpoint_dir_path = args.checkpoint_dir_path
            self.checkpoint_file_path = args.checkpoint_dir_path
        else:
            self.checkpoint_dir_path = os.path.dirname(args.checkpoint_dir_path)
            self.checkpoint_file_path = args.checkpoint_dir_path

        self.display_steps = args.display_steps
        self.monitor_metric_list = [
            {'name': n, 'type': t}
            for n, t in zip(args.monitor_metric_name, args.monitor_metric_type)
        ]
        # variables
        self.cur_train_results = None
        self.cur_eval_results = None
        self.cur_best_results = {key: None for key in args.monitor_metric_name}
        self.cur_monitor_metric = None
        self.cur_epoch_id = None
        self.num_eval_steps_per_epoch = None
        self.global_step =  {
            Trainer.TRAIN: 0,
            Trainer.VALIDATION: 0,
            Trainer.TEST: 0,
        }
        self._create_output_dir()
        self.evaluator = Evaluator(
            checkpoint_dir_path=self.checkpoint_dir_path,            
            console_output=False)




    def run(self, mode):
        if mode == Trainer.TRAIN:
            self._before_training()
            self._train()
            #self._test()
            self._after_training()
        if mode == Trainer.TEST:
            self._test()

    def _before_training(self):
        self.tensorboard_writer = self._get_tensorboard_writer()

    def _after_training(self):
        self.tensorboard_writer.close()

    def _train(self):
        '''
        state_dict = torch.load(
            self.checkpoint_dir_path+'/change_insize_best_mlr1.ckpt',
            map_location='cuda:{}'.format(self.gpu))
        self.model.load_state_dict(state_dict)
        '''
        for epoch_id in tqdm(range(self.num_epochs)):
            self.cur_epoch_id = epoch_id
            self.cur_train_results = self._train_epoch()
            #self.cur_eval_results = self._eval_epoch(Trainer.VALIDATION)
            
            self.evaluator.evaluate(
                self.model,
                additional_eval_info='train_epoch{}'.format(self.cur_epoch_id))
            
            torch.save(self.model.state_dict(), self.checkpoint_file_path+'/tr_epoc_{}'.format(self.cur_epoch_id))

    def _test(self):
        for metric in self.monitor_metric_list:
            self.cur_monitor_metric = metric
            for epoch_id in tqdm(range(1)):
                self.cur_epoch_id = epoch_id
                self._load()
                self.evaluator.evaluate(
                    self.model,
                    additional_eval_info='test_epoch{}'.format(self.cur_epoch_id))
                
           

    def _load(self):
        state_dict = torch.load(
            self.checkpoint_file_path,
            map_location='cuda:{}'.format(self.gpu))
        self.model.load_state_dict(state_dict)
        load_info = 'loading checkpoint from: {}'.format(
            self.checkpoint_file_path)
        print(load_info)


    def _train_epoch(self):
        raise NotImplementedError

    def _eval_epoch(self, mode:str):
        raise NotImplementedError

  
    def _create_output_dir(self):
        if not os.path.exists(self.checkpoint_dir_path):
            if os.path.isdir(self.checkpoint_dir_path):
                os.makedirs(self.checkpoint_dir_path)
            else:
                os.makedirs(os.path.dirname(self.checkpoint_dir_path))
                
    def _get_tensorboard_writer(self):
        if os.path.exists(self.tensorboard_log_dir_path):
            shutil.rmtree(self.tensorboard_log_dir_path)
        tensorboard_writer = SummaryWriter(self.tensorboard_log_dir_path)
        return tensorboard_writer

    '''
    def _switch_mode(self, mode):
        if mode == Trainer.TRAIN:
            self.model.train()
            #self.dataset.switch_to_train_data()
        
        elif mode == Trainer.VALIDATION:
            self.model.eval()
            self.dataset.switch_to_val_data()
            self.num_eval_steps_per_epoch = self.num_valid_steps_per_epoch
        
        elif mode == Trainer.TEST:
            self.model.eval()
            self.dataset.switch_to_test_data()
            self.num_eval_steps_per_epoch = self.num_test_steps_per_epoch
        
        else:
            error_info = 'mode "{}" is invalid.'.format(mode)
            raise ValueError(error_info)
    '''

    
    @property
    def tensorboard_log_dir_path(self):
        return os.path.join('./output/71/roberta_pretrain', 'tensorboard_logs/'+TIMESTAMP)

    @property
    def num_train_steps_per_epoch(self):
        return len(self.dataset)

    @property
    def num_total_train_steps(self):
        return self.num_train_steps_per_epoch * self.num_epochs


    #@property
    #def checkpoint_file_path(self):
    #    return self.checkpoint_dir_path
    
