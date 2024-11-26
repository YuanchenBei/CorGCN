import matplotlib.pyplot as plt
import dgl
import numpy as np
import torch
import torch.nn as nn
import random
import argparse
import os
import torch.nn.functional as F
from utils import load_humloc, load_pcg, load_blogcatalog, load_ppi, load_delve
from utils import graph_process, create_split, build_mask, EarlyStopper
from model import CorGCN_Model
from trainers import step_train
from trainers import step_eval
from torch.optim.lr_scheduler import StepLR
import warnings
import time
warnings.filterwarnings("ignore")


# setting the random seed of the running environment.
def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    ####
    # the below line work with the commented code of model.py in lines 68-73 and lines 88-92 will make the results more stable.
    # torch.use_deterministic_algorithms(True)
    ####
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    argparser = argparse.ArgumentParser("GNN training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--use_gpu", action="store_true")
    argparser.add_argument("--gpu_id", type=int, default=0)
    argparser.add_argument("--seed", type=int, default=2024)
    argparser.add_argument("--gnn_layers", type=int, default=2)
    argparser.add_argument("--num_epochs", type=int, default=1500)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--dropout", type=float, default=0.3)
    argparser.add_argument("--train_rate", type=float, default=0.6)
    argparser.add_argument("--val_rate", type=float, default=0.2)
    argparser.add_argument("--weight_decay", type=float, default=1e-6)
    argparser.add_argument("--hidden_dim", type=int, default=64)
    argparser.add_argument("--bs", type=int, default=4096)
    argparser.add_argument("--data_name", type=str, default="dblp")
    argparser.add_argument("--model_name", type=str, default="test")
    argparser.add_argument("--param_path", default="best_param")
    argparser.add_argument("--runs", type=int, default=1)
    argparser.add_argument("--early_epochs", type=int, default=100)
    argparser.add_argument("--att_heads", type=int, default=3) 
    argparser.add_argument("--sage_agg", type=str, default='mean') 
    argparser.add_argument("--gin_agg", type=str, default='sum') 
    argparser.add_argument("--rec_param", type=float, default=50)
    argparser.add_argument("--ali_param", type=float, default=0.1)
    argparser.add_argument("--struct_param", type=float, default=0.1)
    argparser.add_argument("--pres_param", type=float, default=1.0)
    argparser.add_argument("--k_num", type=int, default=5)
    argparser.add_argument("--tau", type=float, default=1.0)
    argparser.add_argument("--cluster_step", type=int, default=20)
    argparser.add_argument("--cluster_num", type=int, default=0)
    args = argparser.parse_args()

    if args.use_gpu:
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    test_hamming_loss, test_ranking_loss, test_ma_f1, test_mi_f1 = [], [], [], []
    test_macro_ap, test_micro_ap = [], []
    test_macro_auc, test_micro_auc = [], []
    test_lrap = []
    for run in range(1, args.runs+1):
        if args.runs == 1:
            set_seed(args.seed)
        else:
            set_seed(run)

        # dataset selection
        if args.data_name == 'humloc':
            graph = load_humloc()
        elif args.data_name == 'pcg':
            graph = load_pcg()
        elif args.data_name == 'blogcatalog':
            graph = load_blogcatalog()
        elif args.data_name == 'ppi':
            graph = load_ppi()
        elif args.data_name == 'delve':
            graph = load_delve()
        else:
            raise Exception("None of the available dataset is selected!")

        # graph loading
        graph = graph_process(graph, to_bidirect=False).to(device)
        graph.ndata['feat_emb'] = torch.zeros((graph.num_nodes(),args.hidden_dim)).to(device)
        feat_dim = graph.ndata['feat'].shape[1]
        n_classes = graph.ndata['label'].shape[1]
        train_idx, val_idx, test_idx = create_split(graph, args.train_rate, args.val_rate)
        graph = build_mask(graph, train_idx, val_idx, test_idx, device=device)
        print(f"feature dimension: {feat_dim}, total classes: {n_classes}")

        # model loading
        model = CorGCN_Model(feat_dim, args.hidden_dim, n_classes, args.k_num, bs=args.bs, dropout=args.dropout, device=device).to(device)

        # basic running environment preparing
        graph_save_name = f'{args.param_path}/{args.model_name}_{args.data_name}_graph.pt'
        batch_idx_path = f'{args.param_path}/{args.model_name}_{args.data_name}_batch_idx.pt'
        early_stopper = EarlyStopper(num_trials=args.early_epochs, save_path=f'{args.param_path}/{args.model_name}_{args.data_name}.pt',
                                        save_graph_path=graph_save_name, batch_idx_path = batch_idx_path)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.5)
        struct_triplet_loss = (nn.TripletMarginWithDistanceLoss(distance_function=
                                                                lambda x, y: 1.0 - F.cosine_similarity(x, y)))

        model.focal_loss_init(graph.ndata['label'], graph.ndata['train_mask'])
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        train_loader = dgl.dataloading.DataLoader(
            graph, torch.LongTensor(torch.arange(graph.number_of_nodes())).to(device), graph_sampler=dgl.dataloading.MultiLayerFullNeighborSampler(args.gnn_layers),
            device=device, num_workers=0, batch_size=args.bs, drop_last=False, shuffle=True
        )

        graph_list = []
        empty_graph = dgl.graph(data=([],[]),num_nodes=graph.num_nodes()).to(device)
        empty_graph.ndata['feat'] = graph.ndata['feat']
        empty_graph.ndata['label'] = graph.ndata['label']
        if args.cluster_num > 0:
            for i in range(args.cluster_num):
                graph_list.append(empty_graph)
        else:
            for i in range(n_classes):
                graph_list.append(empty_graph)
        graph_list.append(graph)
        mp_graph_list = graph_list
        best_cluster_map = None
        cluster_map = None
        is_cluster = False

        # model training
        for epoch in range(args.num_epochs):
            model.train()

            if args.cluster_num > 0 and epoch % args.cluster_step == 0:
                is_cluster = True
            now_loss, mp_graph_list, batch_idx, cluster_map = step_train(model, graph_list, optimizer, scheduler, train_loader, sampler,
                                                            struct_triplet_loss, args.rec_param, args.ali_param, args.pres_param, tau=args.tau, 
                                                            device=device, is_cluster=is_cluster, cluster_num=args.cluster_num, cluster_map=cluster_map)

            eval_hamming_loss, eval_ranking_loss, eval_ma_f1, eval_mi_f1, eval_macro_ap, eval_micro_ap, \
            eval_macro_auc, eval_micro_auc, eval_lrap = step_eval(model, mp_graph_list, batch_idx, sampler, cluster_map, plt, mode='val', device=device)

            print(f'Training loss at epoch {epoch}: {now_loss}')
            print(f'Eval hamming: {eval_hamming_loss}, eval ranking: {eval_ranking_loss}, eval macro f1: {eval_ma_f1}, eval micro f1: {eval_mi_f1}')
            print(f'Eval macro AP: {eval_macro_ap}, eval micro AP: {eval_micro_ap}, eval macro AUC: {eval_macro_auc}, eval micro AUC: {eval_micro_auc}, eval lrap: {eval_lrap}')
            
            if early_stopper.best_micro_auc < eval_micro_auc:
                best_cluster_map = cluster_map
            
            if not early_stopper.is_continuable(model, mp_graph_list, batch_idx, eval_hamming_loss, eval_ranking_loss, eval_ma_f1,
                                                eval_mi_f1, eval_macro_ap, eval_micro_ap, eval_macro_auc, eval_micro_auc, eval_lrap):
                print(f'Train done! Best hamming loss: {early_stopper.best_hamming_loss}, best ranking loss: {early_stopper.best_ranking_loss}, best macro-F1: {early_stopper.best_macro_f1}, best micro-F1: {early_stopper.best_micro_f1}')
                print(f'More metrics, best macro-AP: {early_stopper.best_macro_ap}, best micro-AP: {early_stopper.best_micro_ap}, best macro-AUC: {early_stopper.best_macro_auc}, best micro-AUC: {early_stopper.best_micro_auc}, best lrap: {early_stopper.best_lrap}')
                break
            print("##########################################")

        # loading the best model parameters for testing
        chkpt = torch.load(f'{args.param_path}/{args.model_name}_{args.data_name}.pt', map_location='cpu')
        model.load_state_dict(chkpt)
        model = model.to(device)
        mp_graph_list = torch.load(graph_save_name)
        mp_graph_list = [g.to(device) for g in mp_graph_list]
        batch_idx = torch.load(batch_idx_path)

        # final test results outputing
        eval_hamming_loss_test, eval_ranking_loss_test, eval_ma_f1_test, eval_mi_f1_test, eval_macro_ap_test, eval_micro_ap_test, \
        eval_macro_auc_test, eval_micro_auc_test, eval_lrap_test = step_eval(model, mp_graph_list, batch_idx, sampler, best_cluster_map, plt, mode='test', device=device)

        print("$ Test hamming loss: ", eval_hamming_loss_test)
        print("$ Test ranking loss: ", eval_ranking_loss_test)
        print("$ Test macro-F1: ", eval_ma_f1_test)
        print("$ Test micro-F1: ", eval_mi_f1_test)
        print("$ Test macro-AP: ", eval_macro_ap_test)
        print("$ Test micro-AP: ", eval_micro_ap_test)
        print("$ Test macro-AUC: ", eval_macro_auc_test)
        print("$ Test micro-AUC: ", eval_micro_auc_test)
        print("$ Test LRAP: ", eval_lrap_test)
        test_hamming_loss.append(eval_hamming_loss_test)
        test_ranking_loss.append(eval_ranking_loss_test)
        test_ma_f1.append(eval_ma_f1_test)
        test_mi_f1.append(eval_mi_f1_test)
        test_macro_ap.append(eval_macro_ap_test)
        test_micro_ap.append(eval_micro_ap_test)
        test_macro_auc.append(eval_macro_auc_test)
        test_micro_auc.append(eval_micro_auc_test)
        test_lrap.append(eval_lrap_test)

    hamming_loss_mean, hamming_loss_std = np.mean(np.array(test_hamming_loss)), np.std(np.array(test_hamming_loss))
    ranking_loss_mean, ranking_loss_std = np.mean(np.array(test_ranking_loss)), np.std(np.array(test_ranking_loss))
    ma_f1_mean, ma_f1_std = np.mean(np.array(test_ma_f1)), np.std(np.array(test_ma_f1))
    mi_f1_mean, mi_f1_std = np.mean(np.array(test_mi_f1)), np.std(np.array(test_mi_f1))
    macro_ap_mean, macro_ap_std = np.mean(np.array(test_macro_ap)), np.std(np.array(test_macro_ap))
    micro_ap_mean, micro_ap_std = np.mean(np.array(test_micro_ap)), np.std(np.array(test_micro_ap))
    macro_auc_mean, macro_auc_std = np.mean(np.array(test_macro_auc)), np.std(np.array(test_macro_auc))
    micro_auc_mean, micro_auc_std = np.mean(np.array(test_micro_auc)), np.std(np.array(test_micro_auc))
    lrap_mean, lrap_std = np.mean(np.array(test_lrap)), np.std(np.array(test_lrap))

    print(f"Final test hamming loss performance: {hamming_loss_mean} ± {hamming_loss_std}")
    print(f"Final test ranking loss performance: {ranking_loss_mean} ± {ranking_loss_std}")
    print(f"Final test macro-F1 performance: {ma_f1_mean} ± {ma_f1_std}")
    print(f"Final test micro-F1 performance: {mi_f1_mean} ± {mi_f1_std}")
    print(f"Final test macro-AP performance: {macro_ap_mean} ± {macro_ap_std}")
    print(f"Final test micro-AP performance: {micro_ap_mean} ± {micro_ap_std}")
    print(f"Final test macro-AUC performance: {macro_auc_mean} ± {macro_auc_std}")
    print(f"Final test micro-AUC performance: {micro_auc_mean} ± {micro_auc_std}")
    print(f"Final test LRAP performance: {lrap_mean} ± {lrap_std}")


main()
