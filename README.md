# CorGCN
This is the source code of CorGCN (Correlation-Aware Graph Convolutional Networks for Multi-Label Node Classification). The adopted GNN backbone in our source code is GCN.

## Requirements
Since the code depends on *older versions* of DGL, PyG, and Torch, these libraries and their dependencies need to be installed via *wheel files*. *If your newer versions can run the code successfully, feel free to use them directly.* DGL, however, tends to have compatibility issues in newer versions, so version 0.8 is likely required.
```
python >= 3.8
torch  == 1.11.0
torch-geometric == 2.4.0
dgl == 0.8.1
scikit-learn == 1.3.0
pandas == 1.5.3
numpy == 1.23.1
matplotlib
tqdm
```

Below is the method for installing older versions of PyG, DGL, and Torch with other dependencies using wheel files (you can skip this step if it's not needed for your environment).
```
# torch
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# pyg wheel page: https://pytorch-geometric.com/whl/torch-1.11.0%2Bcu113.html
# Firstly, you need to manually download the following dependency libraries from the page and then install them.
pip install pyg_lib-0.1.0+pt111cu113-cp38-cp38-linux_x86_64.whl
pip install torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.15-cp38-cp38-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
# Then, installing the pyg library:
pip install torch-geometric==2.4.0

# dgl
pip install dgl-cu113==0.8.1 dglgo==0.0.1 -f https://data.dgl.ai/wheels/repo.html

# others
pip install scikit-learn==1.3.0 pandas==1.5.3 numpy==1.23.1 matplotlib tqdm
```


## Dataset Access
We have provided the experimental datasets in the ./dataset folder for direct usage. The large-scale Delve dataset is provided in the anonymized Google Cloud Disk due to the too-large folder size: https://drive.google.com/file/d/1PI7xAE03v3TC2VpVpYMtbeaNRSevwyt5/view?usp=sharing, which you should download, unzip, and place it into the ./dataset folder.

The python file utils.py also contains the corresponding processing code of each dataset.

## Experiment Running
Firstly, you should create a new folder in the source code's root folder.
```
mkdir best_param
```

Then, we provide all the running scripts and corresponding datasets to reproduce our experimental results. Specifically, we provide an overall running script file **run.sh** in ./script folder, which can run directly in bash.
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
