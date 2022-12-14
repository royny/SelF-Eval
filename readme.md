Implementation of COLING 2022 Oral presentation paper ["SelF-Eval: Self-supervised Fine-grained Dialogue Evaluation"](https://arxiv.org/abs/2208.08094).

## Prerequisites
First create an environment:
```
conda create -n self python=3.6
```

Then install the required packages:
```
pip install -r requirements.txt
```

Install Texar locally:
```
cd texar-pytorch
pip install .
```

## Training

When training, make sure the CHECKPOINT_DIR_PATH key in pretrain.sh is a directory path. This will be the path where we store our checkpoints and experiments.

```
sh pretrain.sh
```

## Testing

First change the MODE key to test in the file pretrain.sh. Then specify the path to the checkpoint at CHECKPOINT_DIR_PATH. Note, different from training phase, the key now needs to be the path to the checkpoint file. Finally run:

```
sh pretrain.sh
```

