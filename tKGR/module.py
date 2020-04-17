import logging
import time

import numpy as np
import pdb
import torch

import torch.nn as nn

from utils import get_segment_ids


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        attn = self.dropout(attn)  # [n * b, l_v, d]

        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)

        # self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        nn.init.normal_(self.w_node_transform.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()

        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_node_transform(q).view(sz_b, len_q, n_head, d_k)

        k = self.w_node_transform(k).view(sz_b, len_k, n_head, d_k)

        v = self.w_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2)  # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3])  # [(n*b), lq, lk, dk]

        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1)  # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3])  # [(n*b), lq, lk, dk]

        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x lq x lk

        ## Map based Attention
        # output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3)  # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3)  # [(n*b), lq, lk]

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        attn = self.dropout(attn)  # [n * b, l_q, l_k]

        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn


def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    '''
    This class implemented the Bochner's time embedding
    '''

    def __init__(self, expand_dim, device='cpu'):
        '''

        :param expand_dim: number of samples draw from p(w), which are used to estimate kernel based on MCMC
        refer to Self-attention with Functional Time Representation Learning for more detail
        '''
        super(TimeEncode, self).__init__()

        time_dim = expand_dim
        self.basis_freq = torch.nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim)).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts):
        '''

        :param ts: [batch_size, seq_len]
        :return: [batch_size, seq_len, time_dim]
        '''
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = torch.unsqueeze(ts, dim=2)
        # print("Forward in TimeEncode: ts is on ", ts.get_device())
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [batch_size, seq_len, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)
        return harmonic


class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """

    def __init__(self, feat_dim, edge_dim, time_dim,
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.time_dim = time_dim

        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim
        # self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        # self.act = torch.nn.ReLU()

        assert (self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head,
                                                        d_model=self.model_dim,
                                                        d_k=self.model_dim // n_head,
                                                        d_v=self.model_dim // n_head,
                                                        dropout=drop_out)
            self.logger.info('Using scaled prod attention')

        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head,
                                                                d_model=self.model_dim,
                                                                d_k=self.model_dim // n_head,
                                                                d_v=self.model_dim // n_head,
                                                                dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        '''

        :param src: float Tensor of shape [B, D]
        :param src_t: float Tensor of shape [B, 1, Dt], Dt == D
        :param seq: float Tensor of shape [B, N, D]
        :param seq_t: float Tensor of shape [B, N, Dt]
        :param seq_e: float Tensor of shape [B, N, De], De == D
        :param mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.
        :return:
        output, weight
        output: float Tensor of shape [B, D]
        weight: float Tensor of shape [B, N]
        '''
        src_ext = torch.unsqueeze(src, dim=1)  # src [B, 1, D]
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_e, seq_t], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]

        mask = torch.unsqueeze(mask, dim=2)  # mask [B, N, 1]
        mask = mask.permute([0, 2, 1])  # mask [B, 1, N]

        # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask)  # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze(dim=1)
        attn = attn.squeeze(dim=1)

        output = self.merger(output, src)
        return output, attn


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class TGAN(torch.nn.Module):
    def __init__(self, ngh_finder, num_nodes=None, num_edges=None, embed_dim=None, n_feat=None, e_feat=None,
                 attn_mode='prod', use_time='time', agg_method='attn',
                 num_layers=3, n_head=4, null_idx=0, drop_out=0.1, seq_len=None, device='cpu',
                 looking_afterwards=False):
        """
        initialize TGAN, either (num_nodes, num_edge, embed_dim) are given or (n_feat, e_feat) are given.
        If (n_feat, e_feat) are given, this pre-trained embedding is being used.
        :param ngh_finder: an instance of NeighborFinder
        :param num_nodes:
        :param num_edges:
        :param embed_dim: dimension of node/edge embedding
        :param n_feat: numpy array of node embedding, [num_nodes+1, feature_dim]
        :param e_feat: numpy array of edge embedding
        :param attn_mode: attention method
        :param use_time: use time embedding
        :param agg_method: aggregation method
        :param num_layers:
        :param n_head: number of multihead
        :param null_idx: use null_idx to represent dummy node when there is fewer neighbors than required
        :param drop_out:
        :param seq_len:
        :param looking_afterwards: if (object, timestamp) from events happening later can also be neighbors,
        more specifically, use get_temporal_neighbor or get_temporal_neighrbor_v2
        """
        super(TGAN, self).__init__()
        assert num_nodes is not None or n_feat is not None
        assert num_edges is not None or e_feat is not None
        assert embed_dim is not None or n_feat is not None

        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx

        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.looking_afterwards = looking_afterwards

        self.logger = logging.getLogger(__name__)

        if n_feat is not None:
            self.n_feat_th = torch.nn.Parameter(
                torch.from_numpy(n_feat.astype(np.float32)))  # len(n_feat_th) = num_enitiy + 1
            self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=False).cpu()
            self.feat_dim = self.n_feat_th.shape[1]
        else:
            self.node_raw_embed = torch.nn.Embedding(num_nodes + 1, embed_dim).cpu()
            nn.init.xavier_normal_(self.node_raw_embed.weight)
            self.feat_dim = embed_dim
        if e_feat is not None:
            self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32))).cpu()
            self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=False)
        else:
            self.edge_raw_embed = torch.nn.Embedding(num_edges + 1, embed_dim).cpu()
            nn.init.xavier_normal_(self.edge_raw_embed.weight)

        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim

        self.use_time = use_time
        self.device = device

        self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
                                                              self.feat_dim,
                                                              self.feat_dim,
                                                              attn_mode=attn_mode,
                                                              n_head=n_head,
                                                              drop_out=drop_out) for _ in range(num_layers)])

        self.time_encoder = TimeEncode(expand_dim=self.n_feat_dim, device=device)

        self.hidden_target_proj = torch.nn.Linear(2 * embed_dim, embed_dim)

    # def link_predict(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
    #     """
    #     predict the probability of link exists between entity src_idx_l and entity target_idx_l at time cut_time_l
    #     :param src_idx_l: numpy array of subject index [batch_size, ]
    #     :param target_idx_l: numpy array of object index [batch_size, ]
    #     :param cut_time_l: numpy array of timestamp [batch_size, ]
    #     :param num_neighbors: int
    #     :return:
    #     score: tensor [batch, num_relation]
    #     """
    #     src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)  # [batch_size, feature_dim]
    #     target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers,
    #                                  num_neighbors)  # [batch_size, feature_dim]
    #     rel_embed = self.edge_raw_embed(torch.from_numpy(np.arange(self.num_edges))).long().to(
    #         self.device)  # [Num_relations, feature_dim]
    #     rel_embed = torch.unsqueeze(rel_embed, 0)  # [1, num_relations, feature_dim]

    #     # TBD: inference using s(t),p(t),o
    #     score_temp = torch.unsqueeze(src_embed, 1) * rel_embed * torch.unsqueeze(target_embed,
    #                                                                              1)  # [batch_size, num_relation, feature_dim]
    #     score = torch.sum(score_temp, dim=2, keepdim=False)  # [batch_size, num_relation]
    #     return score

    def get_node_emb(self, node_idx_l, device):
        """
        help function to get node embedding
        self.node_raw_embed[0] is the embedding for dummy node, i.e. node non-existing

        Arguments:
            node_idx_l {np.array} -- indices of nodes
        """
        return self.node_raw_embed(torch.from_numpy(node_idx_l + 1).long()).to(device)

    def get_rel_emb(self, rel_idx_l, device):
        """
        help function to get relation embedding
        self.edge_raw_embed[0] is the embedding for dummy relation, i.e. relation non-existing
        Arguments:
            rel_idx_l {[type]} -- [description]
        """
        return self.edge_raw_embed(torch.from_numpy(rel_idx_l + 1).long()).to(device)

    def obj_predict(self, src_idx, rel_idx, cut_time, obj_candidate=None, num_neighbors=20, eval_batch=128):
        """
        predict the probability distribution of all objects given (s, p, t)
        :param obj_candidate: 1-d list of index of candidate objects for which score are calculated, if None use all objects
        :param src_idx: int
        :param rel_idx: int
        :param cut_time: int
        :param num_neighbors: int
        :param eval_batch: how many objects are fed to calculate the score
        :return:
        obj_score: tensor [num_entity, ]
        """
        src_idx_l = np.array([src_idx])
        rel_idx_l = np.array([rel_idx])
        cut_time_l = np.array([cut_time])

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)  # tensor [1, feature_dim]
        rel_embed = self.get_rel_emb(rel_idx_l, self.device)  # tensor [1, feature_dim]

        if obj_candidate is None:
            obj_candidate = np.arange(self.num_nodes)
        else:
            obj_candidate = np.array(obj_candidate)

        num_entity = len(obj_candidate)

        # too much entity, split them
        target_embed_list = []
        for target_start_index in np.arange(0, num_entity, eval_batch):
            target_batch_l = obj_candidate[target_start_index:target_start_index + eval_batch]
            cut_time_repeat = np.repeat(cut_time_l, len(target_batch_l))
            target_embed = self.tem_conv(target_batch_l, cut_time_repeat,
                                         self.num_layers, num_neighbors)  # tensor [num_entity, feature_dim]
            target_embed_list.append(target_embed)

        target_embed = torch.cat(target_embed_list, dim=0)
        obj_score = torch.sum(src_embed * rel_embed * target_embed, dim=1, keepdim=False)  # [num_entity]
        return obj_score

    def forward(self, src_idx, target_idx, neg_idx, cut_time, num_neighbors=20):
        """
        :param src_idx: numpy array of subject index [batch_size, ]
        :param target_idx: numpy array of object index [batch_size, ]
        :param neg_idx: numpy array of false object index, [batch_size, num_neg Q],
                        if neg_idx is None, don't calculate embedding for negative sampled nodes
                        return neg_idx_l as None
        :param cut_time: numpy array of timestamp [batch_size, ]
        :param num_neighbors: int
        :return:
        output of encoder, i.e. representation of src_idx_l, target_idx_l and neg_idx_l
        src_idx_l: [batch_size, num_dim]
        target_idx_l: [batch_size, num_dim]
        neg_idx_l: [batch_size, num_neg Q, num_dim]
        """
        batch_size = neg_idx.shape[0]
        num_neg = neg_idx.shape[1]
        query_time = cut_time if self.looking_afterwards else None
        src_embed = self.tem_conv(src_idx, cut_time, self.num_layers, num_neighbors, query_time)
        target_embed = self.tem_conv(target_idx, cut_time, self.num_layers, num_neighbors, query_time)
        neg_idx_flatten = neg_idx.flatten()  # [batch_size x num_neg,]
        # repeat cut_time num_neg times along axis = 0, so that each negative sampling have a cutting time
        if neg_idx is not None:
            num_neg = neg_idx.shape[1]
            cut_time_repeat = np.repeat(cut_time, num_neg, axis=0)  # [batch_size x num_neg, ]
            neg_embed = self.tem_conv(neg_idx_flatten, cut_time_repeat, self.num_layers, num_neighbors)
            neg_embed = neg_embed.view(batch_size, num_neg, -1)
        else:
            neg_embed = None

        return src_embed, target_embed, neg_embed.view(batch_size, num_neg, -1)

    def DistMult_decoder(self, src_embed_t, target_embed_t, rel_embed_t, neg_embed_t):
        '''

        :param rel_embed_t: tensor [batch_size, num_dim]
        :param src_embed_l: tensor [batch_size, num_dim]
        :param target_embed_l: tensor [batch_size, num_dim]
        :param neg_embed_l: tensor [batch_size, num_neg, num_dim]
        :param obj_candidate:
        :return:
        loss: tensor
        '''
        with torch.no_grad():
            pos_label = torch.ones(len(src_embed_t), dtype=torch.float, device=self.device)
            neg_label = torch.zeros(neg_embed_t.shape[0] * neg_embed_t.shape[1], dtype=torch.float, device=self.device)

        pos_score = torch.sum(src_embed_t * rel_embed_t * target_embed_t, dim=1)  # [batch_size, ]
        neg_score = torch.sum(torch.unsqueeze(src_embed_t, 1) * torch.unsqueeze(rel_embed_t, 1) * neg_embed_t,
                              dim=2).view(-1)  # [batch_size x num_neg_sampling, ]

        loss = torch.nn.BCELoss(reduction='sum')(pos_score.sigmoid(), pos_label)
        loss += torch.nn.BCELoss(reduction='sum')(neg_score.sigmoid(), neg_label)
        loss /= len(pos_score) + len(neg_score)
        return loss

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors, query_time_l=None):
        """
        For target node at time t, aggregate features of its neighborhood $\mathcal{N}(v_0; t)={v_1, ..., v_N}$,
        i.e. entities that have interaction with target node prior to t,
        and combined it with its own feature.
        :param num_neighbors: number of neighbors to draw for a source node
        :param src_idx_l: numpy.array, a batch of source node index [batch_size, ]
        :param cut_time_l: numpy.array, a batch of cutting time [batch_size, ]
        :param curr_layers: indicator for recursion
        :return:
        a new feature representation for nodes in src_idx_l at corresponding cutting time [batch_size, dim]
        """
        assert (curr_layers >= 0)

        batch_size = len(src_idx_l)

        if curr_layers == 0:
            return self.get_node_emb(src_idx_l, self.device)
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l,
                                               cut_time_l,
                                               curr_layers=curr_layers - 1,
                                               num_neighbors=num_neighbors,
                                               query_time_l=query_time_l)

            if self.looking_afterwards:
                assert (query_time_l is not None)
                src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch, src_ngh_query_t_batch = self.ngh_finder.get_temporal_neighbor_v2(
                    src_idx_l,
                    cut_time_l,
                    query_time_l,
                    num_neighbors=num_neighbors)
            else:
                src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
                    src_idx_l,
                    cut_time_l,
                    num_neighbors=num_neighbors)
                src_ngh_query_t_batch = None

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(self.device)

            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(self.device)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat,
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1,
                                                   num_neighbors=num_neighbors,
                                                   query_time_l=src_ngh_query_t_batch)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)

            # get edge time features and node features
            # print("src_ngb_t_batch shape: ", src_ngh_t_batch_th.shape)
            # print("src_ngb_t_batch on ", src_ngh_t_batch_th.get_device())
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.get_rel_emb(src_ngh_eidx_batch, self.device)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]

            # query node always has the start time -> time span == 0
            src_node_t_embed = self.time_encoder(torch.zeros(batch_size, 1).to(self.device))

            local, weight = attn_m(src_node_conv_feat,
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   src_ngn_edge_feat,
                                   mask)
            return local

    def temp_conv_debug(self, src_idx_l, cut_time_l):
        hidden_target_node = self.get_node_emb(src_idx_l, self.device)
        hidden_target_time = self.time_encoder(torch.from_numpy(cut_time_l[:, np.newaxis]).to(self.device))
        return self.hidden_target_proj(torch.cat([hidden_target_node, torch.squeeze(hidden_target_time, 1)], axis=1))


class F(torch.nn.Module):
    def __init__(self, input_dims, output_dims, n_layers, name=None):
        super(F, self).__init__(name=name)

        if n_layers == 1:
            self.linears = nn.ModuleList([nn.Linear(input_dims, output_dims),
                                          nn.Tanh()])
            nn.init.xavier_normal_(self.linears[0])
        elif n_layers == 2:
            self.linears = nn.ModuleList([nn.Linear(output_dims, input_dims),
                                          nn.LeakyReLU(),
                                          nn.Linear(output_dims),
                                          nn.Tanh()])

            nn.init.xavier_normal_(self.linears[0].weight)
            nn.init.xavier_normal_(self.linears[2].weight)
        else:
            raise ValueError("Invalid n_layers")

    def forward(self, inputs, training=None):
        """
        concatenate inputs along last dimensions and feed into MLP
        :param inputs[i]: bs x ... x n_dims
        :param training:
        :return:
        """
        x = torch.cat(inputs, axis=-1)
        for l in self.linears:
            x = l(x)
        return x


class G(torch.nn.Module):
    def __init__(self, left_dims, right_dims, output_dims):
        """[summary]
        bilinear mapping along last dimension of x and y:
        output = MLP_1(x)^T A MLP_2(y), where A is two-dimenion matrix

        Arguments:
            left_dims {[type]} -- input dims of MLP_1
            right_dims {[type]} -- input dims of MLP_2
            output_dims {[type]} -- [description]
        """
        super(G, self).__init__()
        self.left_dense = nn.Linear(left_dims, output_dims)
        self.right_dense = nn.Linear(right_dims, output_dims)
        self.center_dense = nn.Linear(output_dims, output_dims)
        self.left_act = nn.LeakyReLU()
        self.right_act = nn.LeakyReLU()

    def forward(self, inputs):
        """[summary]
        Arguments:
            inputs: (left, right)
            left[i] -- tensor, bs x ... x left_dims
            right[i] -- tensor, bs x ... x right_dims
        """
        left, right = inputs
        left_x = torch.cat(left, dim=-1)
        right_x = torch.cat(right, dim=-1)
        # speed of batch-wise dot production: sum over element-wise product > matmul > bmm
        # refer to https://discuss.pytorch.org/t/dot-product-batch-wise/9746/12
        return torch.sum(
            self.left_act(self.left_dense(left_x)) * self.center_dense(self.right_act(self.right_dense(right_x))),
            dim=-1)


def node2edge_v2_op(inputs, selected_edges, return_vi=True, return_vj=True):
    """ inputs (hidden): n_selected_nodes x n_dims
        selected_edges: n_selected_edges x 8 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by idx, vi, vj )
    """
    assert selected_edges is not None
    assert return_vi or return_vj
    hidden = inputs
    result = []
    if return_vi:
        new_idx_e2vi = selected_edges[:, 6]  # n_selected_edges
        hidden_vi = hidden[new_idx_e2vi]  # n_selected_edges x n_dims
        result.append(hidden_vi)
    if return_vj:
        new_idx_e2vj = selected_edges[:, 7]  # n_selected_edges
        hidden_vj = hidden[new_idx_e2vj]  # n_selected_edges x n_dims
        result.append(hidden_vj)
    return result


def segment_softmax_op(logits, segment_ids):
    logits_segment_softmax = logits.clone()
    for segment_id in sorted(set(segment_ids)):
        # segment_mask = segment_ids==segment_id # somehow no grad is generated
        segment_idx = np.where(segment_ids == segment_id)[0]
        logits_max = torch.max(logits[segment_idx])
        logits_diff = logits[segment_idx] - logits_max
        logits_exp = torch.exp(logits_diff)
        logits_expsum = torch.sum(logits_exp)
        logits_norm = logits_exp / logits_expsum
        logits_segment_softmax[segment_idx] = logits_norm
    return logits_segment_softmax


def aggregate_op_node(logits, target_ids, tc):
    """aggregate attention score of same node, i.e. same (eg_idx, v, t)
    aggregation method: sum

    Arguments:
        logits {Tensor} -- attention score
        target_ids {[type]} -- shape len(logits) x 2, (eg_idx, idx_vj_tj)
        tc: time_cost, record time consumption
    Returns:
        logits_seg_sum, Tensor -- logits_seg_sum[i] is the normalized attention score of target_idx i
    """
    # check if target_ids is a list of continuous interger from 0
    # same (e, t) but with different eg_idx have different target_idx
    num_targets = len(set(target_ids[:, 1]))
    num_eg = len(set(target_ids[:, 0]))
    assert np.max(target_ids[:, 1]) + 1 == num_targets
    assert 0 in target_ids[:, 1]
    assert np.max(target_ids[:, 0]) + 1 == num_eg
    assert 0 in target_ids[:, 0]

    device = logits.get_device()
    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device))

    logits_seg_sum = torch.zeros(num_targets, dtype=torch.float32).to(device)
    logits_eg_sum = torch.zeros(num_eg, dtype=torch.float32).to(device)

    # divide each att score with the sum of att score of the same query graph
    t_start = time.time()
    for eg in range(num_eg):
        logits_eg_sum[eg] = torch.sum(logits[target_ids[:, 0] == eg])
    t_eg_sum = time.time()
    logits = logits / torch.gather(logits_eg_sum, 0, torch.from_numpy(target_ids[:, 0]).long().to(device))
    t_norm = time.time()

    logits_len = len(target_ids)
    sparse_index = torch.LongTensor(np.stack([target_ids[:, 1], np.arange(logits_len)]))
    sparse_value = torch.ones(logits_len, dtype=torch.float)
    trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, sparse_value, torch.Size([num_targets, logits_len])).to(device)
    logits_seg_sum = torch.squeeze(torch.sparse.mm(trans_matrix_sparse, logits.unsqueeze(1)))
    # trans_matrix = np.zeros([num_targets, len(target_ids)], dtype=np.float32)
    # for i, target_id in enumerate(target_ids[:, 1]):
    #     trans_matrix[target_id][i] = 1

    # logits_seg_sum = torch.matmul(torch.from_numpy(trans_matrix).to(device), logits)
    t_seg_sum = time.time()

    tc['model']['DP_attn_aggr_eg_sum'] += t_eg_sum - t_start
    tc['model']['DP_attn_aggr_norm'] += t_norm - t_eg_sum
    tc['model']['DP_attn_aggr_seg_sum'] += t_seg_sum - t_norm

    return logits_seg_sum


def _aggregate_op_entity(logits, nodes):
    """aggregate attention score of same entity, i.e. same (eg_idx, v)

    Arguments:
        logits {Tensor} -- attention score
        nodes {numpy.array} -- shape len(logits) x 3, (eg_idx, v, t), sorted by eg_idx, v, t
    return:
        entity_att_score {Tensor}: shape num_entity
        entities: numpy.array -- shape num_entity x 2, (eg_idx, v)
        att_score[i] if the attention score of entities[i]
    """
    device = logits.get_device()
    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device))

    entities, entities_idx = np.unique(nodes[:, :2], axis=0, return_inverse=True)
    entity_att_score = torch.zeros(len(entities), dtype=torch.float32).to(device)

    for i in range(len(entities)):
        entity_att_score[i] = torch.sum(logits[entities_idx == i])

    return entity_att_score, entities


class AttentionFlow(nn.Module):
    def __init__(self, n_dims, n_dims_sm, device='cpu'):
        """[summary]

        Arguments:
            n_dims -- int, dimension of entity and relation embedding
            n_dims_sm -- int, smaller than n_dims to reduce the compuation consumption of calculating attention score
        """
        super(AttentionFlow, self).__init__()
        self.n_dims = n_dims_sm

        self.proj = nn.Linear(n_dims, n_dims_sm)
        self.transition_fn = G(5 * n_dims_sm, 5 * n_dims_sm, n_dims_sm)

        self.device = device

    def get_init_node_attention(self, src_idx_l, cut_time_l):
        """
        return initialized node_attention
        Arguments:
            src_idx_l {numpy.array} [batch_size] -- numpy array of entity index
            cut_time_l {numpy.array} [batch_size] -- numpy array of cut time

        Returns:
            numpy.array {numpy.array} [batch_size x 3] (eg_idx, vi, ts） -- initialized node_attention
            attention_score {Tensor} [batch_size] with values of 1
            eg_idx indicates nodes in subgraph for which query
        """
        eg_idx_l = np.arange(len(src_idx_l), dtype=np.int32)
        att_score = np.ones_like(src_idx_l, dtype=np.float32)

        return np.stack([eg_idx_l, src_idx_l, cut_time_l], axis=1), torch.from_numpy(att_score).to(self.device)

    def forward(self, node_attention, selected_edges=None, memorized_embedding=None, rel_emb=None,
                query_src_emb=None, query_rel_emb=None, query_time_emb=None, training=None, tc=None):
        """calculate attention score

        Arguments:
            node_attention {tensor, num_edges} -- src_attention of selected_edges, node_attention[i] is the attention score
            of (selected_edge[i, 1], selected_edge[i, 2]) in eg_idx==selected_edge[i, 0]

        Keyword Arguments:
            selected_edges {numpy.array, num_edges x 8} -- (eg_idx, vi, ti, vj, tj, rel, idx_eg_vi_ti, idx_eg_vj_tj) (default: {None})
            contain selfloop
            memorized_embedding {dict}: (e, t): emb
            query_src_emb {[type]} -- [description] (default: {None})
            query_rel_emb {[type]} -- [description] (default: {None})
            query_time_emb {[type]} -- [description] (default: {None})
            training {[type]} -- [description] (default: {None})
        return:
            new_node_attention: Tensor, shape: n_new_node
        """
        t_start = time.time()
        query_src_vec = self.proj(query_src_emb)  # batch_size x n_dims_sm
        query_rel_vec = self.proj(query_rel_emb)  # batch_size x n_dims_sm
        query_time_vec = self.proj(query_time_emb)  # batch_size x n_dims_sm

        rel_emb = self.proj(rel_emb)

        hidden_vi = torch.stack([memorized_embedding[(e, t)] for e, t in selected_edges[:, [1, 2]]], dim=0).to(
            self.device)
        hidden_vj = torch.stack([memorized_embedding[(e, t)] for e, t in selected_edges[:, [3, 4]]], dim=0).to(
            self.device)

        hidden_vi = self.proj(hidden_vi)
        hidden_vj = self.proj(hidden_vj)

        t_proj = time.time()

        # [embedding]_repeat is a new tensor which index [embedding] so that it mathes hidden_vi and hidden_vj along dim 0
        # i.e. hidden_vi[i] and hidden_vj[i] is representation of node vi, vj that lie in subgraph corresponding to the query,
        # whose src, rel, time embedding is [embedding]_repeat[i]
        # [embedding] is one of query_src, query_rel, query_time
        query_src_vec_repeat = torch.index_select(query_src_vec, dim=0,
                                                  index=torch.from_numpy(selected_edges[:, 0]).long().to(self.device))
        query_rel_vec_repeat = torch.index_select(query_rel_vec, dim=0,
                                                  index=torch.from_numpy(selected_edges[:, 0]).long().to(self.device))
        query_time_vec_repeat = torch.index_select(query_time_vec, dim=0,
                                                   index=torch.from_numpy(selected_edges[:, 0]).long().to(self.device))

        t_query = time.time()

        transition_logits = self.transition_fn(
            ((hidden_vi, rel_emb, query_src_vec_repeat, query_rel_vec_repeat, query_time_vec_repeat),
             (hidden_vj, rel_emb, query_src_vec_repeat, query_rel_vec_repeat, query_time_vec_repeat)))
        t_transition = time.time()

        attending_node_attention = transition_logits * node_attention
        softmax_node_attention = segment_softmax_op(attending_node_attention, selected_edges[:, 6])
        t_softmax = time.time()

        new_node_attention = aggregate_op_node(softmax_node_attention, selected_edges[:, [0, 7]], tc)

        tc['model']['DP_attn_aggr'] += time.time() - t_softmax
        tc['model']['DP_attn_transition'] += t_transition - t_query
        tc['model']['DP_attn_softmax'] += t_softmax - t_transition
        tc['model']['DP_attn_proj'] += t_proj - t_start
        tc['model']['DP_attn_query'] += t_query - t_proj

        return new_node_attention


class tDPMPN(torch.nn.Module):
    def __init__(self, ngh_finder, num_entity=None, num_rel=None, embed_dim=None, embed_dim_sm=None,
                 attn_mode='prod', use_time='time', agg_method='attn', DP_num_neighbors=40,
                 tgan_num_neighbors=20, tgan_num_layers=2, tgan_n_head=4, null_idx=0, drop_out=0.1, seq_len=None,
                 max_attended_nodes=20, max_attending_nodes=200, device='cpu'):
        """[summary]

        Arguments:
            ngh_finder {[type]} -- an instance of NeighborFinder, find neighbors of a node from temporal KG
            according to TGAN scheme

        Keyword Arguments:
            num_entity {[type]} -- [description] (default: {None})
            num_rel {[type]} -- [description] (default: {None})
            embed_dim {[type]} -- [dimension of DPMPN embedding] (default: {None})
            embed_dim_sm {[type]} -- [smaller dimension of DPMPN embedding] (default: {None})
            attn_mode {str} -- [currently only prod is supported] (default: {'prod'})
            use_time {str} -- [use time embedding] (default: {'time'})
            tgan_num_neighbors {int} -- find number of neighbors for a node
            agg_method {str} -- [description] (default: {'attn'})
            tgan_num_layers {int} -- [description] (default: {2})
            tgan_n_head {int} -- [description] (default: {4})
            null_idx {int} -- [description] (default: {0})
            drop_out {float} -- [description] (default: {0.1})
            seq_len {[type]} -- [description] (default: {None})
            max_attended_nodes {int} -- [max number of nodes in attending-from horizon] (default: {20})
            device {str} -- [description] (default: {'cpu'})
        """
        super(tDPMPN, self).__init__()
        self.DP_num_neighbors = DP_num_neighbors
        self.ngh_finder = ngh_finder
        self.selfloop = num_rel  # index of relation "selfloop", therefore num_edges in TGAN need to be increased by 1
        self.TGAN = TGAN(self.ngh_finder, num_nodes=num_entity, num_edges=num_rel + 1, embed_dim=embed_dim,
                         attn_mode=attn_mode, use_time=use_time, agg_method=agg_method,
                         num_layers=tgan_num_layers, n_head=tgan_n_head, null_idx=null_idx, drop_out=drop_out,
                         seq_len=seq_len, device=device)
        self.att_flow = AttentionFlow(embed_dim, embed_dim_sm, device=device)
        self.max_attended_nodes = max_attended_nodes
        self.tgan_num_neighbors = tgan_num_neighbors
        self.memorized_embedding = dict()
        self.device = device

        self.src_idx_l, self.rel_idx_l = None, None

    def set_init(self, src_idx_l, rel_idx_l, target_idx_l, cut_time_l, batch_i, epoch):
        self.src_idx_l = src_idx_l
        self.rel_idx_l = rel_idx_l
        self.target_idx_l = target_idx_l
        self.cut_time_l = cut_time_l
        self.batch_i = batch_i
        self.epoch = epoch

    def initialize(self):
        """[summary]

        Returns:
            query_src_emb, Tensor -- batch_size x n_dim, embedding of queried source entity
            query_rel_emb, Tensor -- batch_size x n_dim, embedding of queried relation
            query_ts_emb, Tensor -- batch_size x n_dim, embedding of queried timestamp
            attending_nodes, np.array -- n_attending_nodes x 3, (eg_idx, entity_id, ts)
            attending_node_attention, np,array -- n_attending_nodes, (1,)
            memorized_embedding, dict ((entity_id, ts): TGAN_embedding)
        """
        query_src_emb = self.TGAN.get_node_emb(self.src_idx_l, self.device)
        query_rel_emb = self.TGAN.get_rel_emb(self.rel_idx_l, self.device)
        query_ts_emb = self.TGAN.time_encoder(
            torch.from_numpy(self.cut_time_l[:, np.newaxis]).to(torch.float32).to(self.device))
        query_ts_emb = torch.squeeze(query_ts_emb, 1)

        attending_nodes, attending_node_attention = self.att_flow.get_init_node_attention(self.src_idx_l,
                                                                                          self.cut_time_l)
        # refer to https://discuss.pytorch.org/t/feeding-dictionary-of-tensors-to-model-on-gpu/68289
        # attending_node_emb = self.TGAN.temp_conv(self.src_idx_l, self.cut_time_l, curr_layers=2,
        #                                         num_neighbors=self.tgan_num_neighbors, query_time_l=self.cut_time_l)
        attending_node_emb = self.TGAN.temp_conv_debug(self.src_idx_l, self.cut_time_l)
        memorized_embedding = {(src_idx, cut_time): emb for src_idx, cut_time, emb in
                               list(zip(self.src_idx_l, self.cut_time_l, attending_node_emb.to('cpu')))}
        return query_src_emb, query_rel_emb, query_ts_emb, attending_nodes, attending_node_attention, memorized_embedding

    def flow(self, attending_nodes, attending_node_attention, memorized_embedding: dict, query_src_emb, query_rel_emb,
             query_time_emb, tc=None):
        """[summary]

        Arguments:
            attending_nodes {numpy.array} -- num_nodes x3 (eg_idx, entity_id, ts), dtype: numpy.int32
            attending_node_attention {Tensor} -- num_nodes, dtype: torch.float32
            attending_node_emb {Tensor} -- num_nodes x n_dim
            query_src_emb {Tensor} -- batch_size x n_dim, dtype: torch.float32
            query_rel_emb {Tensor} -- batch_size x n_dim, dtype: torch.float32
        return:
            selected_node {numpy.array} -- num_selected x 3 (eg_idx, entity_id, ts) sorted by (eg_idx, entity_id, ts)
            new_node_attention {Tensor} -- num_selected
            so that new_node_attention[i] is the attention of selected_node[i]
            memorized_embedding: dict {(e, t): TGAN_embedding}
        """
        assert (len(attending_nodes) == attending_node_attention.shape[0])

        # Attending-from Horizon of last step
        # attended_nodes: (np.array) n_attended_nodes x 3, (eg_idx, vi, cut_time) sorted
        # attended_nodes_attention: Tensor, n_attended_nodes
        attended_nodes, attended_node_attention = self._topk_att_score(attending_nodes, attending_node_attention,
                                                                       self.max_attended_nodes, tc=tc)

        # Sampling Horizon
        # sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, vi, ti, vj, tj, rel), sorted by eg_idx, vi, ti, tj
        # src_attention: (Tensor) n_sampled_edges, attention score of the source node of sampled edges
        # selfloop is added
        sampled_edges, src_attention = self._get_sampled_edges(attended_nodes, attended_node_attention,
                                                               num_neighbors=self.DP_num_neighbors, tc=tc)
        # print(sampled_edges)

        # selected_edges: (np.array) n_sampled_edges x 8, (eg_idx, vi, ti, vj, tj, rel, idx_eg_vi_ti, idx_eg_vj_tj]
        # sorted by eg_idx, ti, tj
        # selected_nodes: (eg_idx, v, t) sorted by (eg_idx, v, t)
        selected_edges, selected_node = self._get_selected_edges(sampled_edges, tc=tc)
        # print(selected_edges)
        # print(selected_node)

        # get hidden representation from TGAN
        unvisited = [(e, t) not in memorized_embedding.keys() for e, t in sampled_edges[:, [3, 4]]]
        unvisited_nodes = sampled_edges[unvisited][:, [0, 3, 4]]

        if tc:
            t_start = time.time()
        # hidden_target = self.TGAN.tem_conv(unvisited_nodes[:, 1],
        #                                unvisited_nodes[:, 2],
        #                                curr_layers=2,
        #                                num_neighbors=self.tgan_num_neighbors,
        #                                query_time_l=self.cut_time_l[unvisited_nodes[:, 0]])
        hidden_target = self.TGAN.temp_conv_debug(unvisited_nodes[:, 1], unvisited_nodes[:, 2])
        if tc:
            tc['model']['temp_conv'] += time.time() - t_start

        memorized_embedding.update(
            {(unvisited_nodes[i][1], unvisited_nodes[i][2]): hidden_target[i].to(torch.float32).to('cpu') for i in
             range(len(unvisited_nodes))})

        rel_emb = self.TGAN.get_rel_emb(selected_edges[:, 5], self.device)

        new_node_attention = self.att_flow(src_attention,
                                           selected_edges=selected_edges,
                                           memorized_embedding=memorized_embedding,
                                           rel_emb=rel_emb,
                                           query_src_emb=query_src_emb,
                                           query_rel_emb=query_rel_emb,
                                           query_time_emb=query_time_emb, tc=tc)
        # there are events with same (s, o, t) but with different relation, therefore length of new_attending_nodes may be
        # smaller than length of selected_edges
        # print("Sampled {} edges, {} nodes".format(len(sampled_edges), len(selected_node)))
        # for i in range(query_rel_emb.shape[0]):
        #     print("graph of {}-th query ({}, {}, ?, {}) contains {} attending nodes, attention_sum: {}".format(
        #         i,
        #         self.src_idx_l[i],
        #         self.rel_idx_l[i],
        #         self.cut_time_l[i],
        #         sum(selected_node[:, 0] == i),
        #         sum(new_node_attention[selected_node[:, 0] == i])))

        return selected_node, new_node_attention, memorized_embedding

    def get_entity_attn_score(self, logits, nodes, tc=None):
        if tc:
            t_start = time.time()
        entity_attn_score = _aggregate_op_entity(logits, nodes)
        if tc:
            tc['model']['entity_attn'] = time.time() - t_start
        return entity_attn_score

    def _get_sampled_edges(self, attended_nodes, node_attention, num_neighbors: int = 20, tc=None):
        """[summary]
        sample neighbors for attended_nodes from all events happen before attended_nodes
        with strategy specified by ngh_finder, selfloop is added
        Arguments:
            attended_nodes {numpy.array} shape: num_attended_nodes x 3 (eg_idx, vi, ti), dtype int32
            -- [nodes (with time) in attended from horizon, for detail refer to DPMPN paper]
            node_attention {Tensor} shape: num_attended_nodes

        Returns:
            sampled_edges: [np.array] -- [shape: n_sampled_edges x 6, (eg_idx, vi, ti, vj, tj, rel)], sorted by eg_idx, ti, tj (ascending) dtype int32
            src_attention: {Tensor} shape: n_sampled_edges, repeated_attention[i] is the attention score of node (sampeled_edges[i, 1], sampled_edges[i, 2])
            for eg_idx=sampled_edges[i, 0]
        """
        assert (len(attended_nodes) == node_attention.shape[0])
        if tc:
            t_start = time.time()
        src_idx_l = attended_nodes[:, 1]
        cut_time_l = attended_nodes[:, 2]
        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
            src_idx_l,
            cut_time_l,
            num_neighbors=num_neighbors)
        # add selfloop
        src_ngh_node_batch = np.concatenate([src_ngh_node_batch, src_idx_l[:, np.newaxis]], axis=1)
        src_ngh_eidx_batch = np.concatenate(
            [src_ngh_eidx_batch, np.array([[self.selfloop] for _ in range(len(attended_nodes))], dtype=np.int32)],
            axis=1)
        src_ngh_t_batch = np.concatenate([src_ngh_t_batch, cut_time_l[:, np.newaxis]], axis=1)
        # removed padded neighbors, with node idx == rel idx == -1 t == 0
        src_ngh_node_batch_flatten = src_ngh_node_batch.flatten()
        src_ngh_eidx_batch_flatten = src_ngh_eidx_batch.flatten()
        src_ngh_t_batch_faltten = src_ngh_t_batch.flatten()
        eg_idx = np.repeat(attended_nodes[:, 0], num_neighbors + 1)
        mask = src_ngh_node_batch_flatten != -1

        sampled_edges = np.stack([eg_idx,
                                  np.repeat(src_idx_l, num_neighbors + 1), np.repeat(cut_time_l, num_neighbors + 1), \
                                  src_ngh_node_batch_flatten, src_ngh_t_batch_faltten, \
                                  src_ngh_eidx_batch_flatten], axis=1)[mask]

        src_attention = node_attention.view(-1, 1).repeat(1, num_neighbors + 1).view(-1)[mask]

        if tc:
            tc['graph']['sample'] += time.time() - t_start
        return sampled_edges, src_attention

    def _get_selected_edges(self, sampled_edges, tc=None):
        '''
        idx_eg_vi_ti, idx_eg_vj_tj facilitate message aggregation,
        avoiding aggregating message from the same node of another batch
        :param sampled_edges: n_sampled_edges x 6, (eg_idx, vi, ti, vj, tj, rel) sorted by (eg_idx, ti, tj) ascending
        :return:
        selected_edges: (np.array) n_selected_edges (=n_sampled_edges) x 8, (eg_idx, vi, ti, vj, tj, rel, idx_eg_vi_ti, idx_eg_vj_tj]
                        sorted by eg_idx, vi, tj
        selected_nodes: (np.array) n_attending_nodes x 3 (eg_idx, v, t), sorted by (eg_idx, v, t) ascending
        '''
        if tc:
            t_start = time.time()
        if len(sampled_edges) == 0:
            return np.zeros((0, 6), dtype='int32')

        _, idx_eg_vi_ti = np.unique(sampled_edges[:, :3], axis=0, return_inverse=True)
        selected_nodes, idx_eg_vj_tj = np.unique(sampled_edges[:, [0, 3, 4]], axis=0, return_inverse=True)

        idx_eg_vi_ti = np.expand_dims(np.array(idx_eg_vi_ti, dtype='int32'), 1)
        idx_eg_vj_tj = np.expand_dims(np.array(idx_eg_vj_tj, dtype='int32'), 1)

        selected_edges = np.concatenate([sampled_edges[:, :6], idx_eg_vi_ti, idx_eg_vj_tj], axis=1)
        if tc:
            tc['graph']['select'] += time.time() - t_start
        return selected_edges, selected_nodes

    def _get_selfloop_edges(self, attended_nodes):
        eg_idx, vi, ti = attended_nodes[:, 0], attended_nodes[:, 1], attended_nodes[:, 2]
        selfloop_edges = np.stack([eg_idx, vi, ti, vi, ti,
                                   np.repeat(np.array(self.selfloop, dtype='int32'), eg_idx.shape[0])],
                                  axis=1)
        return selfloop_edges  # (eg_idx, vi, ti, vi, ti, selfloop)

    def _get_union_edges(self, selected_edges, repeated_node_attention, selfloop_edges, node_attention):
        """ selected_edges: (np.array) n_selected_edges x 8, (eg_idx, vi, ti, vj, tj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
            selfloop_edges: (np.array) n_selfloop_edges x 6 (eg_idx, vi, ti, vi, ti, selfloop)
        return:
        aug_scanned_edges: (eg_idx, vi, ti, vj, tj, rel, idx_vi, idx_vj)
        aug_node_attention: torch.Tensor
        new_attending_nodes: np.array (eg_idx, vj, tj)
        """
        selected_edges = np.zeros((0, 6), dtype='int32') if len(selected_edges) == 0 else selected_edges[:,
                                                                                          :6]  # (eg_idx, vi, ti, vj, tj, rel)
        all_edges = np.concatenate([selected_edges, selfloop_edges], axis=0).copy()
        all_attention = torch.cat([repeated_node_attention, node_attention], dim=0)
        sorted_idx = np.squeeze(np.argsort(all_edges.view('<i4,<i4,<i4,<i4,<i4,<i4'),
                                           order=['f0', 'f1', 'f3'], axis=0), 1).astype('int32')
        aug_selected_edges = all_edges[sorted_idx]  # sorted by (eg_idx, vi, vj)
        aug_node_attention = all_attention[sorted_idx]
        idx_vi_ti = get_segment_ids(aug_selected_edges[:, :3])
        # new_attending_nodes[idx_vj_tj] --> aug_selected_edges
        new_attending_nodes, idx_vj_tj = np.unique(aug_selected_edges[:, [0, 3, 4]], axis=0, return_inverse=True)
        idx_vi_ti = np.expand_dims(np.array(idx_vi_ti, dtype='int32'), 1)
        idx_vj_tj = np.expand_dims(np.array(idx_vj_tj, dtype='int32'), 1)
        aug_selected_edges = np.concatenate([aug_selected_edges, idx_vi_ti, idx_vj_tj], axis=1)

        return aug_selected_edges, aug_node_attention, new_attending_nodes

    def _add_nodes_to_memorized(self, scanned_edges, hidden_src_emb, hidden_target_emb):
        for i, edge in enumerate(scanned_edges):
            self.memorized_embedding[(edge[1], edge[2])] = hidden_src_emb[i]
            self.memorized_embedding[(edge[3], edge[4])] = hidden_target_emb[i]

    def _topk_att_score(self, attending_nodes, attending_node_attention, k: int, tc=None):
        """

        :param attending_nodes: numpy array, N_visited_nodes x 3 (eg_idx, vi, ts), dtype np.int32
        :param attending_node_attention: tensor, N_visited_nodes, dtype=torch.float32
        :param k: number of nodes in attended-from horizon
        :param attending_node_emb: embedding of attending nodes
        :return:
        attended_nodes, numpy.array, (eg_idx, vi, ts)
        attended_node_attention, tensor, attention_score, same length as attended_nodes
        attended_node_emb, tensor, same length as attended_nodes
        """
        if tc:
            t_start = time.time()
        res_nodes = []
        res_att = []
        for eg_idx in sorted(set(attending_nodes[:, 0])):
            mask = attending_nodes[:, 0] == eg_idx
            masked_nodes = attending_nodes[mask]
            masked_node_attention = attending_node_attention[mask]
            if masked_nodes.shape[0] <= k:
                res_nodes.append(masked_nodes)
                res_att.append(masked_node_attention)
            else:
                topk_node_attention, indices = torch.topk(masked_node_attention, k)
                # pdb.set_trace()
                res_nodes.append(masked_nodes[indices.cpu().numpy()])
                res_att.append(topk_node_attention)
        if tc:
            tc['graph']['topk'] += time.time() - t_start

        return np.concatenate(res_nodes, axis=0), torch.cat(res_att, dim=0)


class tE2GN(torch.nn.Module):
    """
    tE2N stands for temporal Entity and Event Graph Network
    """

    def __init__(self, num_entity: int, num_rel: int, num_steps: int = 2, embed_dim: int = None,
                 num_samples: int = None, max_num_entities: int = None, max_num_events: int = None,
                 null_embedding: bool = True, device: str = 'cpu'):
        """

        :param num_entity: number of entities
        :param num_rel: number of relations
        :param num_steps: how many steps we run dynamic pruned message passing
        :param embed_dim: we use same length of embedding for entity, relation and time
        :param num_samples: how many events we sample for each entity
        :param max_num_entities: max number of entities kept after pruning on entity graph
        :param max_num_events: max number of events kept after pruning on event graph
        :param null_embedding: indicating existence of embedding of null in node and relation embedding
        :param device:
        """
        super(tE2GN, self).__init__()
        self.num_entity = num_entity
        self.num_rel = num_rel
        self.embed_dim = embed_dim

        self.node_raw_embed = torch.nn.Embedding(num_entity + 1, embed_dim).cpu()
        self.rel_raw_embed = torch.nn.Embedding(num_rel + 1, embed_dim).cpu()

        self.num_steps = num_steps
        self.num_samples = num_samples
        self.max_num_entities = max_num_entities
        self.max_num_events = max_num_events

        self.null_embedding = null_embedding
        self.device = device

        # self.att_flow = AttentionFlow(embed_dim, device=device)

    def set_init(self, events):
        """

        :param events: list of quadruplet (sub, pre, obj, timestamp), shape: [batch_size, ]
        :return:
        """
        self.sub_idx_l = np.array([event[0] for event in events])
        self.rel_idx_l = np.array([event[1] for event in events])
        self.obj_idx_l = np.array([event[2] for event in events])
        self.cut_time_l = np.array([event[3] for event in events])

    def _get_ent_emb(self, ent_idx_l, device=None):
        """
        get entity embedding
        :param ent_idx_l: list of entity index
        :param device: use model device if not specified
        :return:
        """
        if device is None:
            device = self.device
        if self.null_embedding:
            return self.node_raw_embed(torch.from_numpy(ent_idx_l + 1).long()).to(device)
        else:
            return self.node_raw_embed(torch.from_numpy(ent_idx_l).long()).to(device)

    def _get_rel_emb(self, rel_idx_l, device=None):
        """
        get relation embedding
        :param rel_idx_l: list of relation index
        :param device: use model device if not specified
        :return:
        """
        if device is None:
            device = self.device
        if self.null_embedding:
            return self.rel_raw_embed(torch.from_numpy(rel_idx_l + 1).long()).to(device)
        else:
            return self.rel_raw_embed(torch.from_numpy(rel_idx_l).long()).to(device)

    def forward(self):
        pass
