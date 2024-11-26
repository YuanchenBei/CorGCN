# CorGCN
This is the source code of CorGCN (Correlation-Aware Graph Convolutional Networks for Multi-Label Node Classification).

## Requirements
```
python >= 3.8
torch  == 1.11.0
torch-geometric == 2.4.0
dgl == 0.8.1
pickle == 0.7.5
scikit-learn == 1.3.0
pandas == 1.5.3
numpy == 1.23.1
tqdm == 4.65.0
```

## Dataset Access
We have provided the experimental datasets in the ./dataset folder for direct usage. The large-scale Delve dataset is provided in the anonymized Google Cloud Disk due to the too-large folder size: https://drive.google.com/file/d/1PI7xAE03v3TC2VpVpYMtbeaNRSevwyt5/view?usp=sharing, which you should download, unzip, and place it into the ./dataset folder.

The python file utils.py also contains the corresponding processing code of each dataset.

## Experiment Running
We provide all the running scripts and corresponding datasets to reproduce our experimental results.

Specifically, we provide an overall running script file **run.sh** in ./script folder, which can run directly in bash.
```
bash run.sh
```

You can also easily reproduce our experiments with the following command for each dataset.
```
# humloc
python main.py --model_name corgcn --data_name humloc --runs 5 --gpu_id 0 --use_gpu --lr 0.001 --bs 1024 --k_num 7

# pcg
python main.py --model_name corgcn --data_name pcg --runs 5 --gpu_id 0 --use_gpu --lr 0.001 --bs 1024 --k_num 19

# blogcatalog
python main.py --model_name corgcn --data_name blogcatalog --runs 5 --gpu_id 0 --use_gpu --lr 0.001 --bs 4096 --k_num 5

# ppi
python main.py --model_name corgcn --data_name ppi --runs 5 --gpu_id 0 --use_gpu --lr 0.02 --bs 4096 --k_num 5 --cluster_num 20

# delve
python main.py --model_name corgcn --data_name delve --runs 5 --gpu_id 0 --use_gpu --lr 0.001 --bs 8192 --k_num 5 --cluster_num 10
```
