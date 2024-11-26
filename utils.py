import torch
import dgl
from scipy.sparse import load_npz
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
import scipy.sparse as sp
import json

# model's earlystopper
class EarlyStopper(object):
    def __init__(self, num_trials, save_path, save_graph_path, batch_idx_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_hamming_loss = 99999999.0
        self.best_ranking_loss = 99999999.0
        self.best_micro_f1 = 0.0
        self.best_macro_f1 = 0.0
        self.best_micro_ap = 0.0
        self.best_macro_ap = 0.0
        self.best_micro_auc = 0.0
        self.best_macro_auc = 0.0
        self.best_lrap = 0.0
        self.save_path = save_path
        self.save_graph_path = save_graph_path
        self.batch_idx_path = batch_idx_path

    def is_continuable(self, model, mp_graph, batch_idx, eval_hamming_loss, eval_ranking_loss, eval_ma_f1, eval_mi_f1, eval_macro_ap, eval_micro_ap, eval_macro_auc, eval_micro_auc, eval_lrap):
        if eval_micro_auc > self.best_micro_auc:
            self.best_hamming_loss = eval_hamming_loss
            self.best_ranking_loss = eval_ranking_loss
            self.best_macro_f1 = eval_ma_f1
            self.best_micro_f1 = eval_mi_f1
            self.best_macro_ap = eval_macro_ap
            self.best_micro_ap = eval_micro_ap
            self.best_macro_auc = eval_macro_auc
            self.best_micro_auc = eval_micro_auc
            self.best_lrap = eval_lrap
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            torch.save(mp_graph, self.save_graph_path)
            torch.save(batch_idx, self.batch_idx_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            self.trial_counter = 0
            return False

# each dataset loading from the source data file
def load_blogcatalog():
    total_nodes = pd.read_csv('./dataset/blogcatalog/data/nodes.csv', header=None).max()[0]
    total_classes = pd.read_csv('./dataset/blogcatalog/data/groups.csv', header=None).max()[0]
    node_belong = pd.read_csv('./dataset/blogcatalog/data/group-edges.csv', header=None, delimiter=',', names=['node', 'group'])
    label_matrix = np.zeros((total_nodes, total_classes), dtype=int)
    for index, row in node_belong.iterrows():
        i = row[0] - 1 
        c = row[1] - 1
        label_matrix[i, c] = 1
    blog_edge = pd.read_csv('./dataset/blogcatalog/data/edges.csv', delimiter=',', header=None, names=['src', 'dst'])
    blog_edge = blog_edge.apply(subtract_one)
    src = blog_edge['src'].to_numpy()
    dst = blog_edge['dst'].to_numpy()
    g = dgl.graph((src, dst))
    g.ndata['feat'] = torch.rand(total_nodes, 100)
    g.ndata['label'] = torch.LongTensor(label_matrix)
    return g


def load_pcg():
    label_data_df = pd.read_csv('./dataset/pcg/labels.csv', header=None, delimiter=',')
    label_data = label_data_df.values
    feat_data_df = pd.read_csv('./dataset/pcg/features.csv', header=None, delimiter=',')
    feat_data = feat_data_df.values
    pcg_edge = pd.read_csv('./dataset/pcg/edges_undir.csv', header=None, delimiter=',', names=['src', 'dst'])
    src = pcg_edge['src'].to_numpy()
    dst = pcg_edge['dst'].to_numpy()
    g = dgl.graph((src, dst))
    g.ndata['feat'] = torch.FloatTensor(feat_data)
    g.ndata['label'] = torch.LongTensor(label_data)
    return g


def load_humloc():
    label_data_df = pd.read_csv('./dataset/humloc/labels.csv', header=None, delimiter=',')
    label_data = label_data_df.values
    print(label_data[0], label_data.shape)
    feat_data_df = pd.read_csv('./dataset/humloc/features.csv', header=None, delimiter=',')
    feat_data = feat_data_df.values
    print(feat_data[0], feat_data.shape)
    humloc_edge = pd.read_csv('./dataset/humloc/edge_list.csv', header=0, delimiter=',')
    src = humloc_edge['prot1'].to_numpy().astype('int64')
    dst = humloc_edge['prot2'].to_numpy().astype('int64')
    g = dgl.graph((src, dst))
    print(g.num_edges())
    g.ndata['feat'] = torch.FloatTensor(feat_data)
    g.ndata['label'] = torch.LongTensor(label_data)
    return g


def load_ppi():
    feats = np.load('./dataset/ppi/feats.npy')
    class_map = json.load(open('./dataset/ppi/class_map.json', 'r'))
    label_mat = np.zeros((feats.shape[0], len(list(class_map["0"]))), dtype=int)
    for k, v in class_map.items():
        label_mat[int(k)] = list(v)
    adj_full = sp.load_npz('./dataset/ppi/adj_full.npz')
    g = dgl.from_scipy(adj_full)
    g.ndata['feat'] = torch.FloatTensor(feats)
    g.ndata['label'] = torch.LongTensor(label_mat)
    return g


def load_delve():
    label_data = load_npz('./dataset/delve/delve_multi2_matrices/label.npz').astype(np.int32).toarray()
    feat_data = load_npz('./dataset/delve/delve_multi2_matrices/lsi_ngram.npz').toarray()
    feat_data = Normalizer().fit_transform(feat_data)
    delve_edge = pd.read_csv('./dataset/delve/delve_multi2.edgelist', delimiter='\t', header=None, names=['src', 'dst'])
    src = delve_edge['src'].to_numpy()
    dst = delve_edge['dst'].to_numpy()
    g = dgl.graph((src, dst))
    g.ndata['feat'] = torch.FloatTensor(feat_data)
    g.ndata['label'] = torch.LongTensor(label_data)
    return g


def get_neighbor_feat(g, device='cpu'):
    '''get the pooled neighborhood feature of center nodes'''
    edge_values = torch.ones(g.number_of_edges(), dtype=torch.float32).to(device)
    source_node = g.edges()[0].unsqueeze(0)
    target_node = g.edges()[1].unsqueeze(0)
    center_feat = g.ndata['feat']
    sparse_adj = torch.sparse.FloatTensor(torch.cat((source_node, target_node), dim=0), edge_values,
                                          torch.Size([g.number_of_nodes(), g.number_of_nodes()])).to(device)
    degree = torch.sparse.sum(sparse_adj, dim=-1).to_dense().unsqueeze(1)
    nei_feat_sparse = torch.sparse.mm(sparse_adj, center_feat)
    nei_feat = nei_feat_sparse/degree
    return nei_feat


def get_labeled_idx(g):
    labeled_data = (g.ndata['label'].sum(dim=-1) >= 1)
    labeled_idx = torch.nonzero(labeled_data).squeeze(-1)
    return labeled_idx


def get_unlabeled_idx(g):
    unlabeled_data = (g.ndata['label'].sum(dim=-1) == 0)
    unlabeled_idx = torch.nonzero(unlabeled_data).squeeze(-1)
    return unlabeled_idx


def subtract_one(x):
    return x - 1


def create_split(g, train_rate, test_rate):
    labeled_nodes = get_labeled_idx(g)
    num_labeled_nodes = labeled_nodes.shape[0]
    num_train = int(num_labeled_nodes * train_rate)
    num_test = int(num_labeled_nodes * test_rate)
    num_val = num_labeled_nodes - num_train - num_test
    print(f"Train size: {num_train}; Valid size: {num_val}; Test size: {num_test}")

    indices = torch.randperm(num_labeled_nodes)

    shuffled_nodes = labeled_nodes[indices]

    train_idx = shuffled_nodes[:num_train]
    test_idx = shuffled_nodes[num_train:num_train + num_test]
    val_idx = shuffled_nodes[num_train + num_test:]
    return train_idx, val_idx, test_idx


def build_mask(graph, train_idx, val_idx, test_idx, device='cpu'):
    graph = graph.to(device)

    train_mask = torch.zeros(graph.number_of_nodes(), dtype=torch.bool).to(device)
    train_mask[train_idx] = True

    val_mask = torch.zeros(graph.number_of_nodes(), dtype=torch.bool).to(device)
    val_mask[val_idx] = True

    test_mask = torch.zeros(graph.number_of_nodes(), dtype=torch.bool).to(device)
    test_mask[test_idx] = True

    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    return graph


def graph_process(graph, to_bidirect=True, self_loop=True):
    if to_bidirect:
        print("To bi-directed graph...")
        graph = dgl.to_bidirected(graph, copy_ndata=True)

    if self_loop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    return graph
