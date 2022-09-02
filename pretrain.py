from opt import*
#import util.main_utils as main_utils
from dataprocess import Leveled_dataset
from Roberta_metric import RoBERTaMetric
from mlr_trainer import mlr_Trainer
from torch.utils.data import DataLoader

import random
import torch
import numpy as np

def set_seed(seed):
    """Fixes randomness to enable reproducibility.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = parse_pretrain_opt()
    set_seed(args.seed)
    model = RoBERTaMetric(args)
    train_dataset = Leveled_dataset(args.data_path, level=args.train_mode)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_dataset = Leveled_dataset(args.data_path+'/dev.json', mode='test')
    val_dataloader = DataLoader(val_dataset, shuffle=False)
    trainer = mlr_Trainer(model, train_dataloader, val_dataloader, args)
    trainer.run(args.mode)
