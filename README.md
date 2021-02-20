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
Training:
```
python train.py --warm_start_time 48 --emb_dim 256 128 64 32 --batch_size 128 --lr 0.0002 --dataset ICEWS14_forecasting --epoch 20 --sampling 3 --device 0 --DP_steps 3 --DP_num_edges 15 --max_attended_edges 40 --node_score_aggregation sum --ent_score_aggregation sum --mongo
```
Evaluation:
```
python eval.py --load_checkpoint checkpoints_2020_8_20_21_31_48/checkpoint_4.pt --mongo --device 0
```
different num_neighbors can be specified by adding
```
--DP_num_neighbors 30 --max_attended_edges 40
```
store intermediate results by adding 
```
--explainability_analysis
```

--mongo can store configurations and results into mongodb, it's not necessary.
