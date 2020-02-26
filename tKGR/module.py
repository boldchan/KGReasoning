import logging

import numpy as np
import torch

import torch.nn as nn


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
        self.basis_freq = torch.nn.Parameter(torch.from_numpy(1/10 ** np.linspace(0, 9, time_dim)).float())
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
        map_ts = ts * self.basis_freq.view(1,1,-1) # [batch_size, seq_len, time_dim]
        map_ts += self.phase.view(1,1,-1)

        harmonic = torch.cos(map_ts)
        return harmonic


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()

        time_dim = expand_dim
        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        device = ts.device
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim

        self.att_dim = feat_dim  # + time_dim if use_time else feat_dim

        self.act = torch.nn.ReLU()

        self.lstm = torch.nn.LSTM(input_size=2 * self.att_dim,
                                  hidden_size=self.att_dim,
                                  num_layers=1,
                                  batch_first=True)

        self.merge_map = torch.nn.Linear(2 * self.att_dim, self.feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        num_ngh = seq.shape[1]

        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2)

        _, (hn, cn) = self.lstm(seq_x)

        hn = hn.squeeze(dim=0)
        state = torch.cat([src_x, hn], dim=1)
        output = self.act(self.merge_map(state))
        return output, None


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim):
        super(MeanPool, self).__init__()
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merge_map = torch.nn.Linear(3 * self.feat_dim, self.feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        num_ngh = seq.shape[1]

        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2)
        hn = seq_x.mean(dim=1)
        state = torch.cat([src_x, hn], dim=1)
        output = self.act(self.merge_map(state))

        return output, None


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
        output = output.squeeze()
        attn = attn.squeeze()

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
    def __init__(self, ngh_finder, n_feat, e_feat,
                 attn_mode='prod', use_time='time', agg_method='attn',
                 num_layers=3, n_head=4, null_idx=0, drop_out=0.1, seq_len=None, device='cpu'):
        '''

        :param ngh_finder: an instance of NeighborFinder
        :param n_feat: numpy array of node embedding,
        :param e_feat: numpy array of edge embedding
        :param attn_mode: attention method
        :param use_time: use time embedding
        :param agg_method: aggregation method
        :param num_layers:
        :param n_head: number of multihead
        :param null_idx:
        :param drop_out:
        :param seq_len:
        '''
        super(TGAN, self).__init__()
        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx

        self.logger = logging.getLogger(__name__)

        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=False)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=False)

        self.feat_dim = self.n_feat_th.shape[1]

        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim

        self.use_time = use_time
        self.device = device
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)

        if agg_method == 'attn':
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
                                                                  self.feat_dim,
                                                                  self.feat_dim,
                                                                  attn_mode=attn_mode,
                                                                  n_head=n_head,
                                                                  drop_out=drop_out) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim) for _ in range(num_layers)])
        else:
            raise ValueError('invalid agg_method value')

        if use_time == 'time':
            self.time_encoder = TimeEncode(expand_dim=self.n_feat_dim, device=device)
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.n_feat_th.shape[1], seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('invalid time option')

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)  # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)

    def link_predict(self, src_idx_l, rel_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        '''
        predict the probability of link exists between entity src_idx_l and entity target_idx_l at time cut_time_l
        :param src_idx_l: numpy array of subject index [batch_size, ]
        :param rel_idx_l: numpy array of predicate index [batch_size, ]
        :param target_idx_l: numpy array of object index [batch_size, ]
        :param cut_time_l: numpy array of timestamp [batch_size, ]
        :param num_neighbors: int
        :return:
        score: tensor [batch]
        '''
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)  # [batch_size, feature_dim]
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)  # [batch_size, feature_dim]
        rel_embed = self.edge_raw_embed(torch.from_numpy(rel_idx_l)).long().to(self.device)  # [batch_size, feature_dim]

        # TBD: inference using s(t),p(t),o
        score = torch.sum(src_embed*target_embed*rel_embed, dim=1, keepdim=False)
        return score

    def forward(self, src_idx, target_idx, neg_idx, cut_time, num_neighbors=20):
        '''
        :param src_idx: numpy array of subject index [batch_size, ]
        :param target_idx: numpy array of object index [batch_size, ]
        :param neg_idx: numpy array of false object index, [batch_size, num_neg Q]
        :param cut_time: numpy array of timestamp [batch_size, ]
        :param num_neighbors: int
        :return:
        output of encoder, i.e. representation of src_idx_l, target_idx_l and neg_idx_l
        src_idx_l: [batch_size, num_dim]
        target_idx_l: [batch_size, num_dim]
        neg_idx_l: [batch_size, num_neg Q, num_dim]
        '''
        batch_size = neg_idx.shape[0]
        num_neg =neg_idx.shape[1]

        src_embed = self.tem_conv(src_idx, cut_time, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx, cut_time, self.num_layers, num_neighbors)
        neg_idx_flatten = neg_idx.flatten()  # [batch_size x num_neg,]
        # repeat cut_time num_neg times along axis = 0, so that each negative sampling have a cutting time
        cut_time_repeat = np.repeat(cut_time, num_neg, axis=0)  # [batch_size x num_neg, ]

        neg_embed = self.tem_conv(neg_idx_flatten, cut_time_repeat, self.num_layers, num_neighbors)
        return src_embed, target_embed, neg_embed.view(batch_size, num_neg, -1)

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors):
        '''
        For target node at time t, aggregate features of its neighborhood $\mathcal{N}(v_0; t)={v_1, ..., v_N}$,
        i.e. entities that have interaction with target node prior to t,
        and combined it with its own feature.
        :param src_idx: a batch of source node index [batch_size, ]
        :param cut_time: a batch of cutting time [batch_size, ]
        :param curr_layers: indicator for recursion
        :param num_neighbors: number of neighbors to draw for a source node
        :return: a new feature representation for nodes in src_idx_l at corresponding cutting time
        '''
        assert(curr_layers>=0)

        device = self.n_feat_th.device

        batch_size = len(src_idx_l)
        # print(cut_time_l)
        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(self.device)
        cut_time_l_th = torch.from_numpy(cut_time_l).long().to(self.device)
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # print(cut_time_l_th.shape)
        # print('cut_time_l_th in ', cut_time_l_th.get_device())

        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th+1)

        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l,
                                               cut_time_l,
                                               curr_layers=curr_layers - 1,
                                               num_neighbors=num_neighbors)

            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
                src_idx_l,
                cut_time_l,
                num_neighbors=num_neighbors)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(self.device)
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(self.device)

            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat,
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1,
                                                   num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)

            # get edge time features and node features
            # print("src_ngb_t_batch shape: ", src_ngh_t_batch_th.shape)
            # print("src_ngb_t_batch on ", src_ngh_t_batch_th.get_device())
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]

            local, weight = attn_m(src_node_conv_feat,
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   src_ngn_edge_feat,
                                   mask)
            return local
