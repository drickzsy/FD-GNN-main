## Installation

### Envirment

- Python 3.9+

- PyTorch 2.2.2+

- CUDA 11.8+

Bash

```
# 克隆仓库
git clone https://github.com/YourUsername/FD-GNN.git
cd FD-GNN

# 安装依赖
pip install -r requirements.txt
```

## Dataset

Please download from [HOSS ReID Dataset](https://zenodo.org/records/15860212) , and set the root as the following form：

```
data/HOSS/
├── train/
│   ├── optical/
│   └── sar/
├── query/
└── gallery/
```

#### Training

### Pretraining

We utilize 4 GPUs for pretraining

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 6667 train_pair.py --config_file configs/pretrian_transoss.yml MODEL.DIST_TRAIN True
```

### Fine-tune

Single GPU fine-tuning

```bash
python train.py --config_file configs/hoss_fdgnn.yml
```

## Evaluation

```bash
python test.py --config_file configs/hoss_fdgnn.yml \
MODEL.DEVICE_ID "('0')" \
TEST.WEIGHT 'weights/FD_GNN_best.pth'## Acknowledgement
```

Codebase from [Hoss-ReID](https://github.com/Alioth/Hoss-ReID).