
from tqdm import tqdm
import torch
import sklearn.metrics as metric
import dgl
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def get_relative_idxs(nei_idxs, anc_idxs):
    board = torch.zeros(nei_idxs.max() + 1).long()
    mark = torch.arange(len(nei_idxs))
    board[nei_idxs] = mark
    relative_idxs = board[anc_idxs]
    return relative_idxs

# each training step under batch processing
def step_train(model, graph_list, optimizer, scheduler, train_loader, sampler, struct_loss_func, rec_param, ali_param, 
                             pres_param, tau=1.0, device='cpu', is_cluster=False, cluster_num=0, cluster_map=None):
    model.train()
    total_loss = 0.0
    loss_func = nn.BCELoss()
    tk_vis = tqdm(train_loader, smoothing=0, mininterval=1.0)
    mp_graph_list = graph_list
    batch_idx = []
    for i, (input_nodes, output_nodes, blocks) in enumerate(tk_vis):
        blocks = [b.to(device) for b in blocks]
        node_feats = blocks[-1].srcdata['feat']
        node_labels = blocks[-1].srcdata['label'].float()
        node_mask = blocks[-1].srcdata['train_mask']
        cmi_loss, lm_loss, feat_emb, label_emb = model.feature_decomposition(node_feats, node_labels, node_mask, tau)

        if i == 0 and is_cluster:
            _, cluster_map = model.label2cluster(label_emb, cluster_num)
            cluster_map = cluster_map.to(device)
        if cluster_map is not None:
            label_emb = cluster_map.matmul(label_emb)

        nei_idxs = blocks[-1].srcdata[dgl.NID]
        anc_idxs = blocks[-1].dstdata[dgl.NID]
        anc_relative_idxs = get_relative_idxs(nei_idxs, anc_idxs)
        batch_idx.append(anc_idxs)
        
        mp_graph_list = model.structure_decomposition(mp_graph_list, feat_emb, label_emb, nei_idxs, anc_idxs, anc_relative_idxs)     
        cls_results = model.mp_label_cls(mp_graph_list, sampler, label_emb, anc_idxs)
        loss_cls = loss_func(cls_results[node_mask[anc_relative_idxs]], node_labels[anc_relative_idxs][node_mask[anc_relative_idxs]].float())

        with torch.no_grad():
            lm_param = loss_cls / 3.0 / lm_loss
            cmi_param = loss_cls / 3.0 / cmi_loss

        loss = loss_cls + lm_param * lm_loss + cmi_param * cmi_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss, mp_graph_list, batch_idx, cluster_map


# each testing step under batch processing
def step_eval(model, mp_graph_list, batch_idx, sampler, cluster_map, plt, mode, device='cpu'):
    
    model.eval()
    with torch.no_grad():
        eval_label_results = torch.LongTensor([]).to(device)
        eval_prob_results = torch.FloatTensor([]).to(device)
        true_results = torch.LongTensor([]).to(device)

        all_labels = torch.arange(model.n_classes).to(device)  
        label_emb = model.label_encoding(all_labels) 
        if cluster_map is not None:
            label_emb = cluster_map.matmul(label_emb)

        for anc_idxs in batch_idx:
            if mode == 'val':
                node_mask = mp_graph_list[-1].ndata['val_mask']
            else:
                node_mask = mp_graph_list[-1].ndata['test_mask']
            
            node_labels = mp_graph_list[-1].ndata['label'][anc_idxs][node_mask[anc_idxs]]
            cls_results = model.mp_label_cls(mp_graph_list, sampler, label_emb, anc_idxs)
            output_preds = cls_results[node_mask[anc_idxs]]

            output_pred2cate = (output_preds > 0.5).long()
            true_results = torch.cat([true_results, node_labels], dim=0)
            eval_label_results = torch.cat([eval_label_results, output_pred2cate], dim=0)
            eval_prob_results = torch.cat([eval_prob_results, output_preds], dim=0)

        true_results = true_results.cpu().numpy()
        eval_label_results = eval_label_results.cpu().numpy()
        eval_prob_results = eval_prob_results.cpu().numpy()
        return get_metrics(true_results, eval_label_results, eval_prob_results)


def get_metrics(true_results, pred_label_results, pred_prob_results):
    # obtaining the multi-label testing performance
    hamming_loss = metric.hamming_loss(true_results, pred_label_results)
    ranking_loss = metric.label_ranking_loss(true_results, pred_prob_results)
    macro_f1 = metric.f1_score(true_results, pred_label_results, average='macro')
    micro_f1 = metric.f1_score(true_results, pred_label_results, average='micro')
    macro_ap = metric.average_precision_score(true_results, pred_prob_results, average='macro')
    micro_ap = metric.average_precision_score(true_results, pred_prob_results, average='micro')
    macro_auc = metric.roc_auc_score(true_results, pred_prob_results, average='macro')
    micro_auc = metric.roc_auc_score(true_results, pred_prob_results, average='micro')
    lrap = metric.label_ranking_average_precision_score(true_results, pred_prob_results)
    return hamming_loss, ranking_loss, macro_f1, micro_f1, macro_ap, micro_ap, macro_auc, micro_auc, lrap
