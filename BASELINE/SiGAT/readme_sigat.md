# Signed Graph Attention Networks 

Update: We optimized the architecture of the model and proposed a new model (SDGNN), and related research results are published on AAAI2021. [more detail](./readme_sdgnn.md)


## Overview
This [paper](https://arxiv.org/abs/1906.10958) is accepted at ICANN2019.

<div align=center>
 <img src="./imgs/SiGAT.png" alt="Sigat" align=center/>
</div>

> We provide a Pytorch implementation of Signed Graph Attention Networks, which incorporates graph motifs into GAT to capture two well-known theories in signed network research, i.e., balance theory and status theory.

## Requirements

The script has been tested running under Python 3.6.3, with the following packages installed (along with their dependencies):

```
pip install -r requirements.txt
```


## Parameters

```
parser.add_argument('--devices', type=str, default='cpu', help='Devices')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dataset', default='bitcoin_alpha', help='Dataset')
parser.add_argument('--dim', type=int, default=20, help='Embedding Dimension')
parser.add_argument('--fea_dim', type=int, default=20, help='Feature Embedding Dimension')
parser.add_argument('--batch_size', type=int, default=500, help='Batch Size')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout k')
parser.add_argument('--k', default=1, help='Folder k')
```


## Run Example


Firstly, run ```python sigat.py``` get node embeddings, then run ```python logistic_function.py``` to get results.

```
pos_ratio: 0.9394377842083506
accuracy: 0.944605208763952
f1_score: 0.971001947630383
macro f1_score: 0.6767452134465279
micro f1_score: 0.944605208763952
auc score: 0.8886568520333262
```


## Bibtex
Please cite our paper if you use this code in your own work:

```
@inproceedings{huang2019signed,
  title={Signed graph attention networks},
  author={Huang, Junjie and Shen, Huawei and Hou, Liang and Cheng, Xueqi},
  booktitle={International Conference on Artificial Neural Networks},
  pages={566--577},
  year={2019},
  organization={Springer}
}
```

## Acknowledgement

> Some codes are adapted from [paper](https://dl.acm.org/citation.cfm?id=1772756) and [pyGAT](https://github.com/Diego999/pyGAT)
