# KGReasoning

## Installation

- clone repository
- install virtualenv
```
pip install virtualenv
```
- create virtual environment
```
virtualenv -p python3 venv
```
- activate virtualenv
```
source venv/bin/activate
```
- install packages
```
pip install -r requirements.txt
```
- specify directory to save Checkpoint
```
cd tKGR
vim local_config.py
```
For example if you want to save checkpoints where local_config.py is

#### **`local_config.py`**
```python
from pathlib import Path

save_dir = Path(__file__).parent.absolute()
```
#### Run:
Model with multiple individual layers is on branch AttentionLayer
Model with shared layer on branch SharedAttention

checkout to branch you want:
```
git checkout -b BranchName origin/BranchName
```
CUDA_VISIBLE_DEVICES=0 python train.py --warm_start_time 48 --emb_dim 256 --emb_dim_sm 64 --batch_size 128 --lr 0.0002 --dataset ICEWS14_forecasting --epoch 20 --sampling 3 --device 0 --DP_steps 3 --DP_num_neighbors 15 --max_attended_nodes 40 --emb_static_temporal_ratio 2
