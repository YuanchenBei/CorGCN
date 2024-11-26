cd ..

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
