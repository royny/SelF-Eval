Implementation of COLING 2022 Oral presentation paper ["SelF-Eval: Self-supervised Fine-grained Dialogue Evaluation"](https://arxiv.org/abs/2208.08094).

## Prerequisites

Install the required packages:
```
pip install -r requirements.txt
```

Install Texar locally:
```
cd texar-pytorch
pip install .
```

## Training

```
sh pretrain.sh
```

## Testing

First change the MODE key to test in the file pretrain.sh. Then specify the path to the checkpoint at CHECKPOINT_DIR_PATH. Finally run:

```
sh pretrain.sh
```

