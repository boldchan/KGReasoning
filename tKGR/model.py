import os
import sys
import time
import pdb
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)

from segment import segment_softmax_op_v2, segment_topk


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
    expand_dim: int, dimension of temporal entity embeddings
    enitity_specific: bool, whether use entith specific freuency and phase.
    num_entities: number of entities.
    '''

    def __init__(self, expand_dim, entity_specific=False, num_entities=None, device='cpu'):
        """
        :param expand_dim: number of samples draw from p(w), which are used to estimate kernel based on MCMC
        :param entity_specific: if use entity specific time embedding
        :param num_entities: number of entities
        refer to Self-attention with Functional Time Representation Learning for more detail
        """
        super(TimeEncode, self).__init__()
        self.time_dim = expand_dim
        self.entity_specific = entity_specific

        if entity_specific:
            self.basis_freq = torch.nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float().unsqueeze(dim=0).repeat(num_entities, 1))
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float().unsqueeze(dim=0).repeat(num_entities, 1))
        else:
            self.basis_freq = torch.nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float()) # shape: num_entities * time_dim
            self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, ts, entities = None):
        '''
        :param ts: [batch_size, seq_len]
        :param entities: which entities do we extract their time embeddings.
        :return: [batch_size, seq_len, time_dim]
        '''
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = torch.unsqueeze(ts, dim=2)
        # print("Forward in TimeEncode: ts is on ", ts.get_device())
        if self.entity_specific:
            map_ts = ts * self.basis_freq[entities].unsqueeze(dim=1)  # self.basis_freq[entities]:  [batch_size, time_dim]
            map_ts += self.phase[entities].unsqueeze(dim=1)
        else:
            map_ts = ts * self.basis_freq.view(1, 1, -1)  # [batch_size, 1, time_dim]
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
    def __init__(self, n_dims, n_dims_sm, recalculate_att_after_prun, static_embed_dim, temporal_embed_dim, node_score_aggregation='sum', device='cpu'):
        """[summary]

        Arguments:
            n_dims -- int, dimension of entity and relation embedding
            n_dims_sm -- int, smaller than n_dims to reduce the compuation consumption of calculating attention score
        """
        super(AttentionFlow, self).__init__()

        self.proj = nn.Linear(n_dims, n_dims_sm)
        self.static_embed_dims_sm = int(static_embed_dim * n_dims_sm / n_dims)
        self.temporal_embed_dims_sm = int(temporal_embed_dim * n_dims_sm / n_dims)
        self.proj_static_embed = nn.Linear(static_embed_dim, self.static_embed_dims_sm)
        self.proj_temporal_embed = nn.Linear(temporal_embed_dim, self.temporal_embed_dims_sm)
        self.transition_fn = G(3 * n_dims_sm + self.static_embed_dims_sm + self.temporal_embed_dims_sm, 3 * n_dims_sm + \
                               self.static_embed_dims_sm + self.temporal_embed_dims_sm, n_dims_sm)
        # self.linears = nn.ModuleList([nn.Linear(n_dims, n_dims) for i in range(DP_steps)])
        self.linear = nn.Linear(n_dims, n_dims)  # use shared Linear for representation update  #TODO what we mentioned right?
        self.recalculate_att_after_prun = recalculate_att_after_prun
        self.node_score_aggregation = node_score_aggregation

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
        att_score = np.ones_like(src_idx_l, dtype=np.float32)*(1-1e-8)

        return np.stack([eg_idx_l, src_idx_l, cut_time_l, np.arange(len(src_idx_l))], axis=1), torch.from_numpy(att_score).to(self.device)

    def _topk_att_score(self, edges, logits, k: int, tc=None):
        """

        :param edges: numpy array, (eg_idx, vi, ti, vj, tj, rel, node_idx_i, node_idx_j), dtype np.int32
        :param logits: tensor, same length as edges, dtype=torch.float32
        :param k: number of nodes in attended-from horizon
        :return:
        pruned_edges, numpy.array, (eg_idx, vi, ts)
        pruned_logits, tensor, same length as pruned_edges
        """
        if tc:
            t_start = time.time()
        res_edges = []
        res_logits = []
        res_indices = []
        for eg_idx in sorted(set(edges[:, 0])):
            mask = edges[:, 0] == eg_idx
            orig_indices = np.arange(len(edges))[mask]
            masked_edges = edges[mask]
            masked_edges_logits = logits[mask]
            if masked_edges.shape[0] <= k:
                res_edges.append(masked_edges)
                res_logits.append(masked_edges_logits)
                res_indices.append(orig_indices)
            else:
                topk_edges_logits, indices = torch.topk(masked_edges_logits, k)
                res_indices.append(orig_indices[indices.cpu().numpy()])
                # pdb.set_trace()
                try:
                    res_edges.append(masked_edges[indices.cpu().numpy()])
                except Exception as e:
                    print(indices.cpu().numpy())
                    print(max(indices.cpu().numpy()))
                    print(str(e))
                    raise KeyError
                res_logits.append(topk_edges_logits)
        if tc:
            tc['graph']['topk'] += time.time() - t_start

        return np.concatenate(res_edges, axis=0), torch.cat(res_logits, dim=0), np.concatenate(res_indices, axis=0)

    def forward(self, attended_nodes, node_attention, selected_edges_l=None, memorized_embedding=None, rel_emb_l=None,
                query_src_emb=None, query_rel_emb=None, query_time_emb=None, max_edges=10, tc=None):
        """calculate attention score

        Arguments:
            node_attention {tensor, num_edges} -- src_attention of selected_edges, node_attention[i] is the attention score
            of (selected_edge[i, 1], selected_edge[i, 2]) in eg_idx==selected_edge[i, 0]

        Keyword Arguments:
            selected_edges {numpy.array, num_edges x 8} -- (eg_idx, vi, ti, vj, tj, rel, idx_eg_vi_ti, idx_eg_vj_tj) (default: {None})
            contain selfloop
            memorized_embedding torch.Tensor,
            query_src_emb {[type]} -- [description] (default: {None})
            query_rel_emb {[type]} -- [description] (default: {None})
            query_time_emb {[type]} -- [description] (default: {None})
            training {[type]} -- [description] (default: {None})
        return:
            new_node_attention: Tensor, shape: n_new_node
        """
        if tc:
            t_start = time.time()
        query_src_vec = self.proj_static_embed(query_src_emb)  # batch_size x n_dims_sm #TODO
        query_rel_vec = self.proj(query_rel_emb)  # batch_size x n_dims_sm
        query_time_vec = self.proj_temporal_embed(query_time_emb)  # batch_size x n_dims_sm

        rel_emb = self.proj(rel_emb_l[-1])

        hidden_vi_orig = memorized_embedding[selected_edges_l[-1][:,-2]]
        hidden_vj_orig = memorized_embedding[selected_edges_l[-1][:,-1]]
        # hidden_vi_orig = torch.stack([memorized_embedding[(eg, e, t)] for eg, e, t in selected_edges[:, [0, 1, 2]]],
        #                              dim=0).to(
        #     self.device)
        # hidden_vj_orig = torch.stack([memorized_embedding[(eg, e, t)] for eg, e, t in selected_edges[:, [0, 3, 4]]],
        #                              dim=0).to(
        #     self.device)

        hidden_vi = self.proj(hidden_vi_orig)
        hidden_vj = self.proj(hidden_vj_orig)

        if tc:
            t_proj = time.time()

        # [embedding]_repeat is a new tensor which index [embedding] so that it mathes hidden_vi and hidden_vj along dim 0
        # i.e. hidden_vi[i] and hidden_vj[i] is representation of node vi, vj that lie in subgraph corresponding to the query,
        # whose src, rel, time embedding is [embedding]_repeat[i]
        # [embedding] is one of query_src, query_rel, query_time
        query_src_vec_repeat = torch.index_select(query_src_vec, dim=0,
                                                  index=torch.from_numpy(selected_edges_l[-1][:, 0]).long().to(self.device))
        query_rel_vec_repeat = torch.index_select(query_rel_vec, dim=0,
                                                  index=torch.from_numpy(selected_edges_l[-1][:, 0]).long().to(self.device))
        query_time_vec_repeat = torch.index_select(query_time_vec, dim=0,
                                                   index=torch.from_numpy(selected_edges_l[-1][:, 0]).long().to(self.device))

        if tc:
            t_query = time.time()

        transition_logits = self.transition_fn(
            ((hidden_vi, rel_emb, query_src_vec_repeat, query_rel_vec_repeat, query_time_vec_repeat),
             (hidden_vj, rel_emb, query_src_vec_repeat, query_rel_vec_repeat, query_time_vec_repeat)))
        t_transition = time.time()

        # prune edges whose target node attention score is small
        # get source attention score
        src_att = node_attention[selected_edges_l[-1][:, -2]]
        # target_att = transition_logits*src_att
        transition_logits_softmax = segment_softmax_op_v2(transition_logits, selected_edges_l[-1][:, -2], tc=tc)
        target_att = transition_logits_softmax*src_att
#        print("target node score:")
#        print(target_att)
        pruned_edges, pruned_att, orig_indices = self._topk_att_score(selected_edges_l[-1], target_att, max_edges)
#        print("chosen indices:")
#        print(orig_indices)

        # softmax after pruning. remove the pruned nodes from computing graph (check if true and then if it helps)?
        if self.recalculate_att_after_prun:
            transition_logits_pruned = transition_logits[orig_indices]
            transition_logits_pruned_softmax = segment_softmax_op_v2(transition_logits_pruned, pruned_edges[:, 6], tc=tc)
        else:
            transition_logits_pruned_softmax = transition_logits_softmax[orig_indices]

        num_nodes = len(memorized_embedding)
        if self.node_score_aggregation == 'max':
            target_att = transition_logits_pruned_softmax * pruned_att
            max_dict = dict()
            for i in range(len(pruned_edges)):
                score_i = target_att[i].cpu().detach().numpy()
                if score_i > max_dict.get(pruned_edges[i, -1], (0,0))[1]:
                    max_dict[pruned_edges[i, -1]] = (i, score_i)

            # biggest score from all edges (some edges may have the same subject)
            sparse_index = torch.LongTensor(np.stack([np.array(list(max_dict.keys())), np.array([_[0] for _ in max_dict.values()])])).to(self.device)
            trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, torch.ones(len(max_dict)).to(self.device), torch.Size([num_nodes, len(pruned_edges)])).to(self.device)
            attending_node_attention = torch.squeeze(torch.sparse.mm(trans_matrix_sparse, target_att.unsqueeze(1)))
        elif self.node_score_aggregation in ['mean', 'sum']:
            sparse_index = torch.LongTensor(np.stack([pruned_edges[:, 7], np.arange(len(pruned_edges))])).to(self.device)
            trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, transition_logits_pruned_softmax,
                                                       torch.Size([num_nodes, len(pruned_edges)])).to(self.device)
            # expand node attention:
    #        node_attention = torch.zeros(num_nodes).to(self.device).scatter_(0, torch.from_numpy(attended_nodes[:, -1]).to(self.device),
    #                                                         node_attention)
    #        print("expanded node attention", node_attention)
            # ATTENTION: node_attention[i] must be attention of node with node_idx==i
            attending_node_attention = torch.squeeze(torch.sparse.mm(trans_matrix_sparse, pruned_att.unsqueeze(1)))

    #        print("edges for message passing:")
    #        print(pruned_edges[:, [-2, -1]])

            # node score aggregation
            if self.node_score_aggregation == 'mean':
                c = Counter(pruned_edges[:, -1])
                target_node_cnt = torch.tensor([c[_] for _ in pruned_edges[:, -1]]).to(self.device)
                transition_logits_pruned_softmax = torch.div(transition_logits_pruned_softmax, target_node_cnt)
        elif self.node_score_aggregation != 'sum':
            raise ValueError("node score_aggregate can only be mean, sum or max")

        sparse_index_rep = torch.from_numpy(pruned_edges[:, [-2, -1]]).to(torch.int64).to(self.device)
        sparse_index_identical = torch.from_numpy(np.setdiff1d(np.arange(num_nodes), pruned_edges[:, -2])).unsqueeze(1).repeat(1,2).to(self.device)
        sparse_index_rep = torch.cat([sparse_index_rep, sparse_index_identical], axis=0)
        sparse_value = torch.cat([transition_logits_pruned_softmax, torch.ones(len(sparse_index_identical)).to(self.device)])
        trans_matrix_sparse_rep = torch.sparse.FloatTensor(sparse_index_rep.t(), sparse_value, torch.Size([num_nodes, num_nodes])).to(self.device)
        # ATTENTION: memorized_embedding[i] is embedding of node with node_idx==i
        updated_memorized_embedding = torch.sparse.mm(trans_matrix_sparse_rep, memorized_embedding)

        for selected_edges, rel_emb in zip(selected_edges_l[:-1][::-1], rel_emb_l[:-1][::-1]):
            hidden_vi_orig = updated_memorized_embedding[selected_edges[:, -2]]
            hidden_vj_orig = updated_memorized_embedding[selected_edges[:, -1]]

            hidden_vi = self.proj(hidden_vi_orig)
            hidden_vj = self.proj(hidden_vj_orig)

            rel_emb = self.proj(rel_emb)

            query_src_vec_repeat = torch.index_select(query_src_vec, dim=0,
                                                      index=torch.from_numpy(selected_edges[:, 0]).long().to(
                                                          self.device))
            query_rel_vec_repeat = torch.index_select(query_rel_vec, dim=0,
                                                      index=torch.from_numpy(selected_edges[:, 0]).long().to(
                                                          self.device))
            query_time_vec_repeat = torch.index_select(query_time_vec, dim=0,
                                                       index=torch.from_numpy(selected_edges[:, 0]).long().to(
                                                           self.device))
            # pdb.set_trace()
            transition_logits = self.transition_fn(
                ((hidden_vi, rel_emb, query_src_vec_repeat, query_rel_vec_repeat, query_time_vec_repeat),
                 (hidden_vj, rel_emb, query_src_vec_repeat, query_rel_vec_repeat, query_time_vec_repeat)))

#            print("edges for messaage passing:")
#            print(selected_edges[:, [6, 7]])
            sparse_index_rep = torch.from_numpy(selected_edges[:, [-2, -1]]).to(torch.int64).to(self.device)
            sparse_index_identical = torch.from_numpy(np.setdiff1d(np.arange(num_nodes), selected_edges[:, -2])).unsqueeze(1).repeat(1, 2).to(self.device)
            sparse_index_rep = torch.cat([sparse_index_rep, sparse_index_identical], axis=0)
            sparse_value = torch.cat([transition_logits, torch.ones(len(sparse_index_identical)).to(self.device)])
            trans_matrix_sparse_rep = torch.sparse.FloatTensor(sparse_index_rep.t(), sparse_value,
                                                               torch.Size([num_nodes, num_nodes])).to(self.device)
            updated_memorized_embedding = torch.sparse.mm(trans_matrix_sparse_rep, memorized_embedding)

        # new_node_attention = segment_softmax_op_v2(attending_node_attention, selected_node[:, 0], tc=tc) #?
        if tc:
            t_softmax = time.time()


        if tc:
            tc['model']['DP_attn_aggr'] += time.time() - t_softmax
            tc['model']['DP_attn_transition'] += t_transition - t_query
            tc['model']['DP_attn_softmax'] += t_softmax - t_transition
            tc['model']['DP_attn_proj'] += t_proj - t_start
            tc['model']['DP_attn_query'] += t_query - t_proj

        return attending_node_attention, updated_memorized_embedding, pruned_edges, orig_indices


class tDPMPN(torch.nn.Module):
    def __init__(self, ngh_finder, num_entity=None, num_rel=None, emb_dim=None, emb_dim_sm=None,
                 attn_mode='prod', use_time='time', agg_method='attn', DP_num_neighbors=40,
                 null_idx=0, drop_out=0.1, seq_len=None, recalculate_att_after_prun=True,
                 emb_static_ratio=1, diac_embed=False,
                 node_score_aggregation='sum', max_attended_edges=20, device='cpu', **kwargs):
        """[summary]

        Arguments:
            ngh_finder {[type]} -- an instance of NeighborFinder, find neighbors of a node from temporal KG
            according to TGAN scheme

        Keyword Arguments:
            num_entity {[type]} -- [description] (default: {None})
            num_rel {[type]} -- [description] (default: {None})
            emb_dim {[type]} -- [dimension of DPMPN embedding] (default: {None})
            emb_dim_sm {[type]} -- [smaller dimension of DPMPN embedding] (default: {None})
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
        if ngh_finder.sampling == -1:
            assert recalculate_att_after_prun
        self.DP_num_neighbors = DP_num_neighbors
        self.ngh_finder = ngh_finder

        self.temporal_embed_dim = int(emb_dim * 2 / (1 + emb_static_ratio))
        self.static_embed_dim = emb_dim * 2 - self.temporal_embed_dim

        self.entity_raw_embed = torch.nn.Embedding(num_entity, self.static_embed_dim).cpu()
        nn.init.xavier_normal_(self.entity_raw_embed.weight)
        self.relation_raw_embed = torch.nn.Embedding(num_rel + 1, emb_dim).cpu()
        nn.init.xavier_normal_(self.relation_raw_embed.weight)
        self.selfloop = num_rel  # index of relation "selfloop", therefore num_edges in relation_raw_embed need to be increased by 1
        self.att_flow = AttentionFlow(emb_dim, emb_dim_sm, recalculate_att_after_prun=recalculate_att_after_prun,
                                      static_embed_dim = self.static_embed_dim, temporal_embed_dim = self.temporal_embed_dim,
                                      node_score_aggregation=node_score_aggregation, device=device)
        self.max_attended_edges = max_attended_edges

        self.time_encoder = TimeEncode(expand_dim=self.temporal_embed_dim, entity_specific=diac_embed, num_entities=num_entity, device=device)
        self.ent_spec_time_embed = diac_embed
        self.hidden_node_proj = torch.nn.Linear(2 * emb_dim, emb_dim) # project (entity_emb; time_emb) to hidden node embedding

        self.memorized_embedding = dict()
        self.device = device

        self.src_idx_l, self.rel_idx_l = None, None
        self.num_existing_nodes = 0

    def set_init(self, src_idx_l, rel_idx_l, target_idx_l, cut_time_l, batch_i, epoch):
        self.src_idx_l = src_idx_l
        self.rel_idx_l = rel_idx_l
        self.target_idx_l = target_idx_l
        self.cut_time_l = cut_time_l
        self.batch_i = batch_i
        self.epoch = epoch
        self.sampled_edges_l = []
        self.rel_emb_l = []

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
        query_src_emb = self.get_ent_emb(self.src_idx_l, self.device) #TODO: difference to ent_emb? module.py line 624/625?
        query_rel_emb = self.get_rel_emb(self.rel_idx_l, self.device)
        if self.ent_spec_time_embed:
            query_ts_emb = self.time_encoder(
                torch.from_numpy(self.cut_time_l[:, np.newaxis]).to(torch.float32).to(self.device),
                entities=self.src_idx_l)
        else:
            query_ts_emb = self.time_encoder(
            torch.from_numpy(self.cut_time_l[:, np.newaxis]).to(torch.float32).to(self.device))
        query_ts_emb = torch.squeeze(query_ts_emb, 1)

        attending_nodes, attending_node_attention = self.att_flow.get_init_node_attention(self.src_idx_l,
                                                                                          self.cut_time_l)
        # refer to https://discuss.pytorch.org/t/feeding-dictionary-of-tensors-to-model-on-gpu/68289
        # attending_node_emb = self.TGAN.temp_conv(self.src_idx_l, self.cut_time_l, curr_layers=2,
        #                                         num_neighbors=self.tgan_num_neighbors, query_time_l=self.cut_time_l)
        ent_emb = self.get_ent_emb(self.src_idx_l, self.device)
        if self.ent_spec_time_embed:
            time_emb = self.time_encoder(torch.from_numpy(self.cut_time_l[:, np.newaxis]).to(self.device), entities=self.src_idx_l)
        else:
            time_emb = self.time_encoder(torch.from_numpy(self.cut_time_l[:, np.newaxis]).to(self.device))
        attending_node_emb = self.hidden_node_proj(torch.cat([ent_emb, torch.squeeze(time_emb, 1)], axis=1))
        # memorized_embedding = {(i, src_idx, cut_time): emb for i, (src_idx, cut_time, emb) in
        #                        enumerate(list(zip(self.src_idx_l, self.cut_time_l, attending_node_emb.to('cpu'))))}
        memorized_embedding = attending_node_emb
        self.num_existing_nodes = len(attending_node_emb)
        return query_src_emb, query_rel_emb, query_ts_emb, attending_nodes, attending_node_attention, memorized_embedding

    def flow(self, attended_nodes, attended_node_attention, memorized_embedding, query_src_emb, query_rel_emb,
             query_time_emb, tc=None):
        """[summary]

        Arguments:
            attended_nodes {numpy.array} -- num_nodes x 4 (eg_idx, entity_id, ts, node_idx), dtype: numpy.int32, sort (eg_idx, ts, entity_id)
            attended_node_attention {Tensor} -- num_nodes, dtype: torch.float32
            query_src_emb {Tensor} -- batch_size x n_dim, dtype: torch.float32
            query_rel_emb {Tensor} -- batch_size x n_dim, dtype: torch.float32
        return:
            selected_node {numpy.array} -- num_selected x 3 (eg_idx, entity_id, ts) sorted by (eg_idx, entity_id, ts)
            new_node_attention {Tensor} -- num_selected
            so that new_node_attention[i] is the attention of selected_node[i]
            memorized_embedding: dict {(e, t): TGAN_embedding}
        """

        # Sampling Horizon
        # sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, vi, ti, vj, tj, rel), sorted by eg_idx, ti, vi, tj, vj, rel
        # src_attention: (Tensor) n_sampled_edges, attention score of the source node of sampled edges
        # selfloop is added
        sampled_edges, new_sampled_nodes = self._get_sampled_edges(attended_nodes, num_neighbors=self.DP_num_neighbors, tc=tc)
#        print("sampled {} edges, sampled {} nodes".format(len(sampled_edges), len(sampled_nodes)))
#        print("sampled edge:")
#        print(sampled_edges)
#        print("sampled nodes", sampled_nodes)
#         new_sampled_nodes_mask = sampled_nodes[:,3]>max(attended_nodes[:, 3])
# #        print("{} new sampled nodes".format(sum(new_sampled_nodes_mask)))
#         new_sampled_nodes = sampled_nodes[new_sampled_nodes_mask]
        new_sampled_nodes_emb = self.get_node_emb(new_sampled_nodes[:, 1], new_sampled_nodes[:, 2])
        new_memorized_embedding = torch.cat([memorized_embedding, new_sampled_nodes_emb], axis=0)
#        pdb.set_trace()
        if len(new_sampled_nodes) == 0:
            assert len(new_memorized_embedding) == self.num_existing_nodes
            assert max(new_sampled_nodes[:, -1]) + 1 == self.num_existing_nodes
            assert max(sampled_edges[:, -1]) < self.num_existing_nodes
#        print("# new memorized embedding: {}".format(len(new_memorized_embedding)))

        self.sampled_edges_l.append(sampled_edges)
        # print(sampled_edges)

        # selected_edges: (np.array) n_sampled_edges x 8, (eg_idx, vi, ti, vj, tj, rel, idx_eg_vi_ti, idx_eg_vj_tj]
        # sorted by eg_idx, ti, tj
        # selected_nodes: (eg_idx, v, t) sorted by (eg_idx, v, t)
        # selected_edges, selected_node = self._get_selected_edges(sampled_edges, tc=tc)
        # print(selected_edges)
        # print(selected_node)

        # # get hidden representation from TGAN
        # unvisited = [(i, e, t) not in memorized_embedding.keys() for i, e, t in sampled_edges[:, [0, 3, 4]]]
        # unvisited_nodes = sampled_edges[unvisited][:, [0, 3, 4]]
        #
        # if tc:
        #     t_start = time.time()
        # hidden_target = self.get_node_emb(unvisited_nodes[:, 1], unvisited_nodes[:, 2])
        # if tc:
        #     tc['model']['temp_conv'] += time.time() - t_start
        #
        # memorized_embedding.update(
        #     {tuple(unvisited_nodes[i]): hidden_target[i].to(torch.float32).to('cpu') for i in
        #      range(len(unvisited_nodes))})

        rel_emb = self.get_rel_emb(sampled_edges[:, 5], self.device)
        self.rel_emb_l.append(rel_emb)

        new_node_attention, updated_memorized_embedding, pruned_edges, orig_indices = self.att_flow(attended_nodes, attended_node_attention,
                                                                        selected_edges_l=self.sampled_edges_l,
                                                                        memorized_embedding=new_memorized_embedding,
                                                                        rel_emb_l=self.rel_emb_l,
                                                                        query_src_emb=query_src_emb,
                                                                        query_rel_emb=query_rel_emb,
                                                                        query_time_emb=query_time_emb,
                                                                        max_edges=self.max_attended_edges, tc=tc)

        assert len(pruned_edges) == len(orig_indices)
#        print("# pruned_edges {}".format(len(pruned_edges)))
        self.sampled_edges_l[-1] = pruned_edges
        self.rel_emb_l[-1] = self.rel_emb_l[-1][orig_indices]
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

        # get pruned nodes
        pruned_nodes = pruned_edges[:, [0, 3, 4, 7]]
        _, indices = np.unique(pruned_edges[:, [0,4,3]], return_index=True, axis=0)
        # pruned_nodes.view('i8,i8,i8,i8').sort(order=['f0', 'f2', 'f1'], axis=0)
        pruned_nodes = pruned_nodes[indices]
#        print("# pruned nodes {}".format(len(pruned_nodes)))
#        print("pruned nodes {}".format(pruned_nodes))
#        print('node attention:', new_node_attention)

        return pruned_nodes, new_node_attention, updated_memorized_embedding

    def get_node_emb(self, src_idx_l, cut_time_l):
        hidden_node = self.get_ent_emb(src_idx_l, self.device)
        if self.ent_spec_time_embed:
            hidden_time = self.time_encoder(torch.from_numpy(cut_time_l[:, np.newaxis]).to(self.device), entities=src_idx_l)
        else:
            hidden_time = self.time_encoder(torch.from_numpy(cut_time_l[:, np.newaxis]).to(self.device))
        return self.hidden_node_proj(torch.cat([hidden_node, torch.squeeze(hidden_time, 1)], axis=1))

    def get_entity_attn_score(self, logits, nodes, tc=None):
        if tc:
            t_start = time.time()
        entity_attn_score, entities = _aggregate_op_entity(logits, nodes)
        if tc:
            tc['model']['entity_attn'] = time.time() - t_start
        return entity_attn_score, entities

    def _get_sampled_edges(self, attended_nodes, num_neighbors: int = 20, tc=None):
        """[summary]
        sample neighbors for attended_nodes from all events happen before attended_nodes
        with strategy specified by ngh_finder, selfloop is added
        Arguments:
            attended_nodes {numpy.array} shape: num_attended_nodes x 4 (eg_idx, vi, ti, node_idx), dtype int32
            -- [nodes (with time) in attended from horizon, for detail refer to DPMPN paper]
            node_attention {Tensor} shape: num_attended_nodes

        Returns:
            sampled_edges: [np.array] -- [shape: n_sampled_edges x 6, (eg_idx, vi, ti, vj, tj, rel)], sorted by eg_idx, ti, vi, tj (ascending), vj, rel dtype int32
            src_attention: {Tensor} shape: n_sampled_edges, repeated_attention[i] is the attention score of node (sampeled_edges[i, 1], sampled_edges[i, 2])
            for eg_idx=sampled_edges[i, 0]
        """
        if tc:
            t_start = time.time()
        src_idx_l = attended_nodes[:, 1]
        cut_time_l = attended_nodes[:, 2]
        node_idx_l = attended_nodes[:, 3]
        tuple2index = {(eg, src, ts):idx for eg, src, ts, idx in attended_nodes}

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
                                  src_ngh_eidx_batch_flatten, np.repeat(node_idx_l, num_neighbors + 1)], axis=1)[mask]

        # index new selected nodes
        target_nodes_index = []
        new_sampled_nodes = []
        for eg, node, edge in sampled_edges[:, [0,3,4]]:
            if (eg, node, edge) in tuple2index:
                target_nodes_index.append(tuple2index[(eg, node, edge)])
            else:
                tuple2index[(eg, node, edge)] = self.num_existing_nodes
                target_nodes_index.append(self.num_existing_nodes)
                new_sampled_nodes.append([eg, node, edge, self.num_existing_nodes])
                self.num_existing_nodes += 1

        sampled_edges = np.concatenate([sampled_edges, np.array(target_nodes_index)[:, np.newaxis]], axis=1)
        # new_sampled_nodes = sampled_edges[:, [0, 3, 4, 7]]
        # sampled_nodes = np.array([[*k, v] for k, v in tuple2index.items()])
        # sampled_nodes.view('i8,i8,i8,i8').sort(order=['f3'], axis=0)
        new_sampled_nodes = sorted(new_sampled_nodes, key=lambda x: x[-1])
        new_sampled_nodes = np.array(new_sampled_nodes)

        if tc:
            tc['graph']['sample'] += time.time() - t_start
        return sampled_edges, new_sampled_nodes


    def _topk_att_score(self, attending_nodes, attending_node_attention, k: int, tc=None):
        """

        :param attending_nodes: numpy array, N_visited_nodes x 4 (eg_idx, vi, ts, node_idx), dtype np.int32
        :param attending_node_attention: tensor, N_all_visited_nodes, dtype=torch.float32
        :param k: number of nodes in attended-from horizon
        :return:
        attended_nodes, numpy.array, (eg_idx, vi, ts)
        attended_node_attention, tensor, attention_score, same length as attended_nodes
        attended_node_emb, tensor, same length as attended_nodes
        """
        if tc:
            t_start = time.time()
        res_nodes = []
        res_att = []
        attending_node_attention = attending_node_attention[torch.from_numpy(attending_nodes[:, 3]).to(torch.int64).to(self.device)]
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
        return self.entity_raw_embed(torch.from_numpy(ent_idx_l).long().to(embed_device)).to(device)

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
        return self.relation_raw_embed(torch.from_numpy(rel_idx_l).long().to(embed_device)).to(device)
