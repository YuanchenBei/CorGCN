import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import scipy.sparse as sp
import math
from sklearn.metrics.pairwise import cosine_similarity
from torch_sparse import SparseTensor
from sklearn.cluster import KMeans


def get_relative_idxs(center_idxs, compare_idxs):
    board = torch.zeros(center_idxs.max() + 1).long()
    mark = torch.arange(len(center_idxs))
    board[center_idxs] = mark
    relative_idxs = board[compare_idxs]
    return relative_idxs


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class CorGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_classes, num_layers, dropout=0.3, activation=F.relu):
        super(CorGCN, self).__init__()


        layers = nn.ModuleList()
        layers.append(
            GCNConv(in_dim, hidden_dim))
        for i in range(num_layers - 2):
            layers.append(GCNConv(hidden_dim, hidden_dim))
        layers.append(GCNConv(hidden_dim, out_dim))
        self.layers_list = layers

        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        self.n_classes = n_classes

        
        self.attention_q = nn.Linear(hidden_dim, hidden_dim)
        self.attention_k = nn.Linear(hidden_dim, hidden_dim)
        self.attention_v = nn.Linear(hidden_dim, hidden_dim)
        

    def forward(self, graph_mp_blocks, graph_feat_emb, ori_feat_emb, label_emb):

        label_emb_q = self.attention_q(label_emb)

        for i, layer in enumerate(self.layers_list):
            temp_emb = []
            
            
            for j in range(label_emb.shape[0]):
                temp_emb.append(layer(graph_feat_emb[j], torch.stack(graph_mp_blocks[j][i].edges())))
            
            # for j in range(label_emb.shape[0]):
            #     n_num = len(graph_mp_blocks[j][i].srcdata[dgl.NID])
            #     e_num = len(graph_mp_blocks[j][i].edges()[0])
            #     adj = SparseTensor(row=graph_mp_blocks[j][i].edges()[0], col=graph_mp_blocks[j][i].edges()[1], 
            #         value=torch.ones(e_num).to(graph_feat_emb.device), sparse_sizes=(n_num, n_num)).to(graph_feat_emb.device)
            #     temp_emb.append(layer(graph_feat_emb[j], adj.t()))


            graph_feat_emb = torch.stack(temp_emb).permute(1,0,2) 
            graph_feat_emb_v = self.attention_v(graph_feat_emb)
            graph_feat_emb_k = self.attention_k(graph_feat_emb) 
           
            at_score = (label_emb_q.matmul(graph_feat_emb_k.transpose(-1,-2)) / math.sqrt(label_emb_q.shape[-1]))
            at_score = F.softmax(at_score, dim=-1) 
            graph_feat_emb = at_score.matmul(graph_feat_emb_v) 
            graph_feat_emb = graph_feat_emb.permute(1,0,2)

            
            ori_feat_emb = layer(ori_feat_emb, torch.stack(graph_mp_blocks[-1][i].edges()))
            
            # n_num = len(graph_mp_blocks[-1][i].srcdata[dgl.NID])
            # e_num = len(graph_mp_blocks[-1][i].edges()[0])
            # adj = SparseTensor(row=graph_mp_blocks[-1][i].edges()[0], col=graph_mp_blocks[-1][i].edges()[1], 
            #     value=torch.ones(e_num).to(graph_feat_emb.device), sparse_sizes=(n_num, n_num)).to(graph_feat_emb.device)
            # ori_feat_emb = layer(ori_feat_emb, adj.t())

            relative_idxs = get_relative_idxs(graph_mp_blocks[-1][i].srcdata[dgl.NID], graph_mp_blocks[-1][i].dstdata[dgl.NID])
            ori_feat_emb = ori_feat_emb[relative_idxs]

            if i != len(self.layers_list) - 1:
                graph_feat_emb = self.activation(graph_feat_emb)
                ori_feat_emb = self.activation(ori_feat_emb)
                graph_feat_emb = self.dropout(graph_feat_emb)
                ori_feat_emb = self.dropout(ori_feat_emb)
        return graph_feat_emb, ori_feat_emb


class CorGCN_Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, k_num, bs=1024, dropout=0.3, device='cpu'):
        super(CorGCN_Model, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.k_num = k_num
        self.device = device
        self.bs = bs
        self.gcn_layers = CorGCN(hidden_dim, hidden_dim, hidden_dim, n_classes, num_layers=2, dropout=dropout)

        self.feat_encoder = nn.Linear(in_dim, hidden_dim)

        self.label_emb = nn.Embedding(n_classes, hidden_dim)
        torch.nn.init.xavier_uniform_(self.label_emb.weight.data)
        self.label_encoder = nn.Linear(hidden_dim, hidden_dim)

        self.cos = nn.CosineSimilarity(dim=-1)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )


    def to(self, device):
        for i in range(len(self.gcn_layers.layers_list)):
            self.gcn_layers.layers_list[i].to(device)
        return super(CorGCN_Model, self).to(device)

    def focal_loss_init(self, labels, mask, weight_type='sqrt'):
        label_freq = labels[mask].sum(axis=0)
        if weight_type == 'dir':
            class_weights = labels.size(0) / (label_freq + 1e-9)
        elif weight_type == 'sqrt':
            class_weights = labels.size(0) / torch.sqrt(label_freq + 1e-9)
        else:
            raise Exception('weight_type error')
        alpha_weights = class_weights / class_weights.sum()
        self.focal_loss = MultiLabelFocalLoss(alpha=alpha_weights)

    def label_encoding(self, labels):
        label_emb = self.label_emb(labels)
        label_encoded = self.label_encoder(label_emb)
        label_encoded = F.normalize(label_encoded, dim=-1)
        return label_encoded

    def feat_encoding(self, x):
        feat_encoded = self.feat_encoder(x)
        return feat_encoded

    def cmi_loss(self, feat_emb, class_embs, input_label, temp=1.0):
        features = feat_emb
        labels = input_label.float()
        n_label = labels.shape[1]
        emb_labels = torch.eye(n_label).to(self.device)
        mask = torch.matmul(labels, emb_labels)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, class_embs.t()),
            temp)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss

    def lm_loss(self, x, y, w_il):
        mlp_x = self.decoder(x)
        z_label = torch.matmul(y.float(), w_il) / y.sum(1, keepdim=True)
        mlp_l = self.decoder(z_label)
        loss1 = self.focal_loss(mlp_x, y.float())
        loss2 = self.focal_loss(mlp_l, y.float())
        return (loss1 + loss2) / 2

    def get_neighbor_feat(self, g, batch_feats, batch_nodes, device='cpu'):
        batched_neighbor_graph = dgl.node_subgraph(g, batch_nodes)
        
        source_node = batched_neighbor_graph.edges()[0].unsqueeze(0)
        target_node = batched_neighbor_graph.edges()[1].unsqueeze(0)
        edge_values = torch.ones(batched_neighbor_graph.number_of_edges(), dtype=torch.float32).to(device)

        sparse_adj = torch.sparse.FloatTensor(torch.cat((target_node, source_node), dim=0), edge_values,
                                              torch.Size([batched_neighbor_graph.number_of_nodes(),
                                                          batched_neighbor_graph.number_of_nodes()]))
        degree = torch.sparse.sum(sparse_adj, dim=-1).to_dense().unsqueeze(1).to(device)
        nei_feat_sparse = torch.sparse.mm(sparse_adj, batch_feats)
        nei_feat = nei_feat_sparse / degree
        return nei_feat

    def generate_mp_graph(self, graph_list, struct_guided_feats, batch_nids):
        # obtaining the decomposed graphs
        z_batch_num = struct_guided_feats.shape[1]
        cos_feats = struct_guided_feats / torch.norm(struct_guided_feats, dim=-1, keepdim=True)
        simi_mat = cos_feats.matmul(cos_feats.transpose(-2, -1))

        diag_idxs = torch.arange(z_batch_num).to(self.device)
        simi_mat[:, diag_idxs, diag_idxs] = -float('inf')
        simi_mat_node_wise = simi_mat.permute(1, 0, 2)  
        top_k_values, top_k_idxs = torch.topk(simi_mat_node_wise, self.k_num, dim=-1)

        new_graph_list = []
        for i in range(len(graph_list)-1):
            refined_graph = graph_list[i]

            node_indices = torch.arange(z_batch_num).view(-1, 1).expand_as(top_k_idxs[:, i]).to(self.device)
            dst_nodes = batch_nids[node_indices.flatten()]
            src_nodes = batch_nids[top_k_idxs[:, i].flatten()]
            new_graph = dgl.add_edges(refined_graph, src_nodes, dst_nodes)
            new_graph_list.append(new_graph)
        
        new_graph_list.append(graph_list[-1])

        return new_graph_list

    def label2cluster(self, label_emb, cluster_num=20):
        # when encounting large label space: micro label -> macro label cluster
        label_num = label_emb.shape[0]

        if isinstance(label_emb, torch.Tensor):
            label_emb = label_emb.cpu().detach().numpy()

        kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(label_emb)

        label2cluster = torch.tensor(kmeans.labels_, dtype=torch.long)
        cluster_centers = torch.tensor(kmeans.cluster_centers_)   

        transformation_matrix = torch.zeros((cluster_num, label_num), dtype=torch.float)
        transformation_matrix[label2cluster, torch.arange(label_num)] = 1
        transformation_matrix = transformation_matrix / transformation_matrix.sum(dim=-1, keepdim=True)
        return cluster_centers, transformation_matrix

    def feature_decomposition(self, feats, input_label, mask, tau):
        all_labels = torch.arange(self.n_classes).to(self.device) 
        label_emb = self.label_encoding(all_labels) 
        feat_emb = self.feat_encoding(feats) 

        cmi_loss = self.cmi_loss(feat_emb[mask], label_emb, input_label[mask], temp=tau)
        lm_loss = self.lm_loss(feat_emb[mask], input_label[mask], label_emb)

        return cmi_loss, lm_loss, feat_emb, label_emb

    def structure_decomposition(self, graph_list, feat_emb, label_emb, nei_idxs, anc_idxs, anc_relative_idxs):
        z_n = feat_emb.shape[0]  
        z_c = label_emb.shape[0] 
        z_d = feat_emb.shape[1] 

        cos_feat_emb = feat_emb / torch.norm(feat_emb, dim=-1, keepdim=True)
        cos_label_emb = label_emb / torch.norm(label_emb, dim=-1, keepdim=True)
        sim_score = (cos_feat_emb.matmul(cos_label_emb.T) + 1) / 2
        sim_score = sim_score.unsqueeze(2).expand(z_n, z_c, z_d)

        proj_feat_emb = sim_score * feat_emb.unsqueeze(1) 
        proj_feat_emb_plained = proj_feat_emb.view(z_n, z_c * z_d)
        nei_proj_feat_emb_plained = self.get_neighbor_feat(graph_list[-1], proj_feat_emb_plained, nei_idxs, self.device)
        nei_proj_feat_emb = nei_proj_feat_emb_plained.view(z_n, z_c, z_d)

        struct_guided_feats = nei_proj_feat_emb.permute(1, 0, 2)
        struct_guided_feats = struct_guided_feats[:,anc_relative_idxs,:] 
        with torch.no_grad():
            mp_graph_list = self.generate_mp_graph(graph_list, struct_guided_feats, anc_idxs)

        return mp_graph_list

    def mp_label_cls(self, mp_graph_list, sampler, label_emb, anc_idxs):
        graph_mp_blocks = []

        for i in range(len(mp_graph_list)):
            _, _, mp_blocks = sampler.sample_blocks(mp_graph_list[i], anc_idxs)
            graph_mp_blocks.append(mp_blocks)

        cos_label_emb = label_emb / torch.norm(label_emb, dim=-1, keepdim=True)
        graph_feat_emb = self.feat_encoding(mp_graph_list[-1].ndata['feat'][anc_idxs])
        cos_feat_emb = graph_feat_emb / torch.norm(graph_feat_emb, dim=-1, keepdim=True)
        at_score = (cos_label_emb.matmul(cos_feat_emb.T) + 1) / 2
        at_score = at_score.unsqueeze(2).expand(label_emb.shape[0], graph_feat_emb.shape[0], graph_feat_emb.shape[1])
        graph_feat_emb = at_score * graph_feat_emb.unsqueeze(0)
        
        ori_feat_emb = self.feat_encoding(graph_mp_blocks[-1][0].srcdata['feat'])
        graph_cls_emb, ori_cls_emb = self.gcn_layers(graph_mp_blocks, graph_feat_emb, ori_feat_emb, label_emb)
        graph_cls_emb = graph_cls_emb.permute(1,0,2) 

        at_score = (self.cos(graph_cls_emb, label_emb) + 1) / 2 
        at_score = at_score / at_score.sum(dim=-1, keepdim=True)

        cls_emb = torch.cat(((graph_cls_emb * at_score.unsqueeze(-1)).sum(dim=1), ori_cls_emb), dim=-1)
        cls_results = F.sigmoid(self.classifier(cls_emb))

        return cls_results
