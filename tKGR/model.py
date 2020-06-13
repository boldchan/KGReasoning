import time

import numpy as np
import torch
from torch import nn

from utils import get_segment_ids
from segment import segment_softmax_op_v2


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

    num_nodes = len(nodes)
    entities, entities_idx = np.unique(nodes[:, :2], axis=0, return_inverse=True)
    sparse_index = torch.LongTensor(np.stack([entities_idx, np.arange(num_nodes)]))
    sparse_value = torch.ones(num_nodes, dtype=torch.float)
    trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                   torch.Size([len(entities), num_nodes])).to(device)
    entity_att_score = torch.squeeze(torch.sparse.mm(trans_matrix_sparse, logits.unsqueeze(1)))
    # entity_att_score = torch.zeros(len(entities), dtype=torch.float32).to(device)

    # for i in range(len(entities)):
    #     entity_att_score[i] = torch.sum(logits[entities_idx == i])

    return entity_att_score, entities


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
        nn.init.normal_(self.left_dense.weight, mean=0, std=np.sqrt(2.0 / (left_dims)))
        self.right_dense = nn.Linear(right_dims, output_dims)
        nn.init.normal_(self.right_dense.weight, mean=0, std=np.sqrt(2.0 / (right_dims)))
        self.center_dense = nn.Linear(output_dims, output_dims)
        nn.init.xavier_normal_(self.center_dense.weight)
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


class AttentionFlow(nn.Module):
    def __init__(self, n_dims, n_dims_sm, device='cpu'):
        """[summary]

        Arguments:
            n_dims -- int, dimension of entity and relation embedding
            n_dims_sm -- int, smaller than n_dims to reduce the compuation consumption of calculating attention score
        """
        super(AttentionFlow, self).__init__()

        self.proj = nn.Linear(n_dims, n_dims_sm)
        self.transition_fn = G(5 * n_dims_sm, 5 * n_dims_sm, n_dims_sm)
        # self.linears = nn.ModuleList([nn.Linear(n_dims, n_dims) for i in range(DP_steps)])
        self.linear = nn.Linear(n_dims, n_dims)  # use shared Linear for representation update

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
                query_src_emb=None, query_rel_emb=None, query_time_emb=None, training=None, tc=None,
                selected_node=None):
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
        if tc:
            t_start = time.time()
        query_src_vec = self.proj(query_src_emb)  # batch_size x n_dims_sm
        query_rel_vec = self.proj(query_rel_emb)  # batch_size x n_dims_sm
        query_time_vec = self.proj(query_time_emb)  # batch_size x n_dims_sm

        rel_emb = self.proj(rel_emb)

        hidden_vi_orig = torch.stack([memorized_embedding[(eg, e, t)] for eg, e, t in selected_edges[:, [0, 1, 2]]],
                                     dim=0).to(
            self.device)
        hidden_vj_orig = torch.stack([memorized_embedding[(eg, e, t)] for eg, e, t in selected_edges[:, [0, 3, 4]]],
                                     dim=0).to(
            self.device)

        hidden_vi = self.proj(hidden_vi_orig)
        hidden_vj = self.proj(hidden_vj_orig)

        if tc:
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

        if tc:
            t_query = time.time()

        transition_logits = self.transition_fn(
            ((hidden_vi, rel_emb, query_src_vec_repeat, query_rel_vec_repeat, query_time_vec_repeat),
             (hidden_vj, rel_emb, query_src_vec_repeat, query_rel_vec_repeat, query_time_vec_repeat)))
        t_transition = time.time()

        transition_logits = segment_softmax_op_v2(transition_logits, selected_edges[:, 6], tc=tc)
        logits_len = len(node_attention)
        sparse_index = torch.LongTensor(np.stack([selected_edges[:, 7], np.arange(logits_len)])).to(self.device)
        trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, transition_logits,
                                                       torch.Size([len(set(selected_edges[:, 7])), logits_len])).to(
            self.device)
        attending_node_attention = torch.squeeze(torch.sparse.mm(trans_matrix_sparse, node_attention.unsqueeze(1)))
        updated_node_representation = torch.sparse.mm(trans_matrix_sparse.transpose(), self.linear(hidden_vj_orig))

        # new_node_attention = segment_softmax_op_v2(attending_node_attention, selected_node[:, 0], tc=tc) #?
        if tc:
            t_softmax = time.time()

        # new_node_attention = aggregate_op_node(softmax_node_attention, selected_edges[:, [0, 7]], tc)
        # new_node_attention, new_node_representation = aggregate_op_node_score_repre(updated_node_representation, softmax_node_attention, selected_edges[:, [0, 7]])
        new_node_representation_dict = {tuple(selected_node[i]): repre for i, repre in
                                        enumerate(updated_node_representation)}
        memorized_embedding.update(new_node_representation_dict)

        if tc:
            tc['model']['DP_attn_aggr'] += time.time() - t_softmax
            tc['model']['DP_attn_transition'] += t_transition - t_query
            tc['model']['DP_attn_softmax'] += t_softmax - t_transition
            tc['model']['DP_attn_proj'] += t_proj - t_start
            tc['model']['DP_attn_query'] += t_query - t_proj

        return new_node_attention


class tDPMPN(torch.nn.Module):
    def __init__(self, ngh_finder, num_entity=None, num_rel=None, embed_dim=None, embed_dim_sm=None,
                 attn_mode='prod', use_time='time', agg_method='attn', DP_num_neighbors=40,
                 null_idx=0, drop_out=0.1, seq_len=None,
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
        self.entity_raw_embed = torch.nn.Embedding(num_entity + 1, embed_dim).cpu()
        nn.init.xavier_normal_(self.entity_raw_embed.weight)
        self.relation_raw_embed = torch.nn.Embedding(num_rel + 1, embed_dim).cpu()
        nn.init.xavier_normal_(self.relation_raw_embed.weight)
        self.selfloop = 0  # index of relation "selfloop", therefore num_edges in relation_raw_embed need to be increased by 1
        self.att_flow = AttentionFlow(embed_dim, embed_dim_sm, device=device)
        self.max_attended_nodes = max_attended_nodes

        self.time_encoder = TimeEncode(expand_dim=embed_dim, device=device)
        self.hidden_target_proj = torch.nn.Linear(2 * embed_dim, embed_dim)

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
        query_src_emb = self.get_ent_emb(self.src_idx_l, self.device)
        query_rel_emb = self.get_rel_emb(self.rel_idx_l, self.device)
        query_ts_emb = self.time_encoder(
            torch.from_numpy(self.cut_time_l[:, np.newaxis]).to(torch.float32).to(self.device))
        query_ts_emb = torch.squeeze(query_ts_emb, 1)

        attending_nodes, attending_node_attention = self.att_flow.get_init_node_attention(self.src_idx_l,
                                                                                          self.cut_time_l)
        # refer to https://discuss.pytorch.org/t/feeding-dictionary-of-tensors-to-model-on-gpu/68289
        # attending_node_emb = self.TGAN.temp_conv(self.src_idx_l, self.cut_time_l, curr_layers=2,
        #                                         num_neighbors=self.tgan_num_neighbors, query_time_l=self.cut_time_l)
        hidden_target_node = self.get_ent_emb(self.src_idx_l, self.device)
        hidden_target_time = self.time_encoder(torch.from_numpy(self.cut_time_l[:, np.newaxis]).to(self.device))
        attending_node_emb = self.hidden_target_proj(torch.cat([hidden_target_node, torch.squeeze(hidden_target_time, 1)], axis=1))
        memorized_embedding = {(i, src_idx, cut_time): emb for i, (src_idx, cut_time, emb) in
                               enumerate(list(zip(self.src_idx_l, self.cut_time_l, attending_node_emb.to('cpu'))))}
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
        unvisited = [(i, e, t) not in memorized_embedding.keys() for i, e, t in sampled_edges[:, [0, 3, 4]]]
        unvisited_nodes = sampled_edges[unvisited][:, [0, 3, 4]]

        if tc:
            t_start = time.time()
        hidden_target = self.get_node_emb(unvisited_nodes[:, 1], unvisited_nodes[:, 2])
        if tc:
            tc['model']['temp_conv'] += time.time() - t_start

        memorized_embedding.update(
            {tuple(unvisited_nodes[i]): hidden_target[i].to(torch.float32).to('cpu') for i in
             range(len(unvisited_nodes))})

        rel_emb = self.get_rel_emb(selected_edges[:, 5], self.device)

        new_node_attention = self.att_flow(src_attention,
                                           selected_edges=selected_edges,
                                           memorized_embedding=memorized_embedding,
                                           rel_emb=rel_emb,
                                           query_src_emb=query_src_emb,
                                           query_rel_emb=query_rel_emb,
                                           query_time_emb=query_time_emb, tc=tc, selected_node=selected_node)
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

        return selected_node, new_node_attention, memorized_embedding, selected_edges

    def get_node_emb(self, src_idx_l, cut_time_l):
        hidden_target_node = self.get_ent_emb(src_idx_l, self.device)
        hidden_target_time = self.time_encoder(torch.from_numpy(cut_time_l[:, np.newaxis]).to(self.device))
        return self.hidden_target_proj(torch.cat([hidden_target_node, torch.squeeze(hidden_target_time, 1)], axis=1))

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
                try:
                    res_nodes.append(masked_nodes[indices.cpu().numpy()])
                except Exception as e:
                    print(indices.cpu().numpy())
                    print(max(indices.cpu().numpy()))
                    print(str(e))
                    raise KeyError
                res_att.append(topk_node_attention)
        if tc:
            tc['graph']['topk'] += time.time() - t_start

        return np.concatenate(res_nodes, axis=0), torch.cat(res_att, dim=0)

    def get_ent_emb(self, ent_idx_l, device):
        """
        help function to get node embedding
        self.entity_raw_embed[0] is the embedding for dummy node, i.e. node non-existing

        Arguments:
            node_idx_l {np.array} -- indices of nodes
        """
        embed_device = next(self.entity_raw_embed.parameters()).get_device()
        if embed_device == -1:
            embed_device = torch.device('cpu')
        else:
            embed_device = torch.device('cuda:{}'.format(embed_device))
        return self.entity_raw_embed(torch.from_numpy(ent_idx_l + 1).long().to(embed_device)).to(device)

    def get_rel_emb(self, rel_idx_l, device):
        """
        help function to get relation embedding
        self.edge_raw_embed[0] is the embedding for dummy relation, i.e. relation non-existing
        Arguments:
            rel_idx_l {[type]} -- [description]
        """
        embed_device = next(self.relation_raw_embed.parameters()).get_device()
        if embed_device == -1:
            embed_device = torch.device('cpu')
        else:
            embed_device = torch.device('cuda:{}'.format(embed_device))
        return self.relation_raw_embed(torch.from_numpy(rel_idx_l + 1).long().to(embed_device)).to(device)