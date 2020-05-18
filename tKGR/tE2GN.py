import os
import sys
from collections import defaultdict
import itertools
import time
import copy
import argparse
import pdb

import numpy as np
np.set_printoptions(edgeitems=10)
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import dgl
import dgl.function as fn

PackageDir = '/home/ubuntu/KGReasoning/tKGR'
sys.path.insert(1, PackageDir)

from utils import Data, save_config
import local_config

save_dir = local_config.save_dir

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def calc_metric(heads, relations, prediction, targets, filter_pool, num_entity):
    hit_1, hit_3, hit_10, mr, mrr = 0., 0., 0., 0., 0.

    n_preds = prediction.shape[0]
    for i in range(n_preds):
        head = heads[i]
        rel = relations[i]
        tar = targets[i]
        pred = prediction[i]
        fil = list(filter_pool[(head, rel)] - {tar})

        sorted_idx = np.argsort(-pred)
        mask = np.logical_not(np.isin(sorted_idx, fil))
        sorted_idx = sorted_idx[mask]

        rank = num_entity if pred[tar] == 0 else np.where(sorted_idx == tar)[0].item() + 1

        if rank <= 1:
            hit_1 += 1
        if rank <= 3:
            hit_3 += 1
        if rank <= 10:
            hit_10 += 1
        mr += rank
        mrr += 1. / rank

    return hit_1, hit_3, hit_10, mr, mrr


def load_checkpoint(model, ent_raw_embed, rel_raw_embed, ts_raw_embed, optimizer, checkpoint_dir):
    if os.path.isfile(checkpoint_dir):
        print("=> loading checkpoint '{}'".format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        ent_raw_embed.load_state_dict(checkpoint['ent_raw_embed_state_dict'])
        rel_raw_embed.load_state_dict(checkpoint['rel_raw_embed_state_dict'])
        ts_raw_embed.load_state_dict(checkpoint['ts_raw_embed_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_dir, checkpoint['epoch']))
    else:
        raise IOError("=> no checkpoint found at '{}'".format(checkpoint_dir))

    return model, optimizer, start_epoch


class MultiHeadGATLayer(nn.Module):
    """
    Multi-Head Attention
    """
    def __init__(self, dim_model, n_head):
        "h: number of heads; dim_model: hidden dimension"
        super(MultiHeadGATLayer, self).__init__()
        self.d_k = 7 * dim_model // n_head # 7*dim_model (sub, rel, obj, time, q_sub, q_rel, q_ts)
        self.d_v = 4 * dim_model // n_head # 4*dim_model (sub, rel, obj, time)
        self.n_head = n_head
        # W_q, W_k, W_v, W_o
        self.attention_func_wq = nn.Linear(7*dim_model, 7*dim_model, bias=False)
        self.attention_func_wk = nn.Linear(7*dim_model, 7*dim_model, bias=False)
        self.attention_func_wv = nn.Linear(4*dim_model, 4*dim_model, bias=False)
        self.attention_func_wo = nn.Linear(4*dim_model, 4*dim_model, bias=False)

    def get(self, x, context, fields='qkv'):
        "Return a dict of queries / keys / values."
        batch_size = x.shape[0]
        ret = {}
        x_cont = torch.cat([x, context], dim=1)
        if 'q' in fields:
            ret['q'] = self.attention_func_wq(x_cont).view(batch_size, self.n_head, self.d_k)
        if 'k' in fields:
            ret['k'] = self.attention_func_wk(x_cont).view(batch_size, self.n_head, self.d_k)
        if 'v' in fields:
            ret['v'] = self.attention_func_wv(x).view(batch_size, self.n_head, self.d_v)
        return ret
    
    def get_o(self, x):
        "get output of the multi-head attention"
        batch_size = x.shape[0]
        return self.attention_func_wo(x.view(batch_size, -1))


import copy
def clones(module, k):
    return nn.ModuleList(
        copy.deepcopy(module) for _ in range(k)
    )

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = torch.nn.LayerNorm(4*layer.size)
        
    def pre_func(self, i, fields='qkv'):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['event_embed']
            context = nodes.data['query_embed']
            return layer.self_attn.get(x, context, fields=fields)

#             norm_x = layer.sublayer[0].norm(x)
#             norm_context = layer.sublayer[1].norm(context)
#             return layer.self_attn.get(norm_x, norm_context, fields=fields)
        return func
    
    def post_func(self, i):
        """
        1, Normalize (softmax denominator) and get output of multi-Head attention
        2, Applying a two layer position-wise feed forward layer on x then add residual connection:
        """
        layer = self.layers[i]
        def func(nodes):
            x, wv, z = nodes.data['event_embed'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[0].dropout(o)
#             x = layer.sublayer[0](x, layer.feed_forward)
            return {'event_embed': x if i < self.N - 1 else self.norm(x)}
#             return {'event_embed': x if i < self.N - 1 else self.norm(x),}
        return func
    
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn # (key, query, value, mask)
        self.feed_forward = feed_forward
        self.sublayer = [SubLayerWrapper(4*size, dropout), SubLayerWrapper(3*size, dropout)]
        
class SubLayerWrapper(nn.Module):
    '''
    The module wraps normalization, dropout, residual connection into one equation:
    sublayerwrapper(sublayer)(x) = x + dropout(sublayer(norm(x)))
    '''
    def __init__(self, size, dropout):
        super(SubLayerWrapper, self).__init__()
        self.norm = torch.nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


# def get_independent_subgraphs(G, SG_nodes_l):
#     SG_list = G.subgraphs(SG_nodes_l)
#     res = []
#     for sg in SG_list:
#         sg_deepcopy = dgl.DGLGraph()
#         sg_deepcopy.add_nodes(sg.number_of_nodes())
#         sg_deepcopy.add_edges(*sg.edges())
#         sg_deepcopy.ndata['_ID'] = sg.ndata['_ID']
#         res.append(sg_deepcopy)
#     return res

# def get_independent_subgraph(G, SG_nodes):
#     sg = G.subgraph(SG_nodes)
#     sg_deepcopy = dgl.DGLGraph()
#     sg_deepcopy.add_nodes(sg.number_of_nodes())
#     sg_deepcopy.add_edges(*sg.edges())
#     sg_deepcopy.ndata['_ID'] = sg.ndata['_ID']
#     return sg_deepcopy
    

class tE2GN(nn.Module):
    def __init__(self, G, encoder, dim_model, h, event_encoder, id2evts, num_entity, num_relation, num_neighbors=10, max_sg_num_nodes=20, device='cpu'):
        super(tE2GN, self).__init__()
        self.G = G
        self.encoder = encoder
        self.h = h
        self.d_k = 7 * dim_model // h
        self.event_encoder = event_encoder # init event embedding by concatenating embeddings of entities, relation and timestamp
        self.id2evts = id2evts
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.num_neighbors = num_neighbors
        self.max_sg_num_nodes = max_sg_num_nodes
        self.device=device
        
    def propagate_attention(self, g, eids):
        # g is DGLGraph contains selfloop
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'), eids)
        g.apply_edges(scaled_exp('score', np.sqrt(self.d_k)), eids)
        # Update node state
        g.send_and_recv(eids,
                        [fn.src_mul_edge('v', 'score', 'v'),
                         fn.copy_e('score', 'score')], # to message
                        [fn.sum('v', 'wv'), fn.sum('score', 'z')])

        g_reverse = dgl.transform.reverse(g)
        for k, v in g.ndata.items():
            g_reverse.ndata[k] = v
        g_reverse.edata['score'] = g.edata['score']
#         print('score', g_reverse.edata['score'].squeeze())

        g_reverse.apply_edges(fn.e_div_u('score', 'z', 'score'), (eids[1], eids[0])) # normalize score
#         print('normed score', g_reverse.edata['score'].squeeze())
        g_reverse.send_and_recv((eids[1], eids[0]),
                        [fn.src_mul_edge('flow_score', 'score', 'flow_score')], # to message
                        [fn.sum('flow_score', 'flow_score')])
        g.ndata['flow_score'] = g_reverse.ndata['flow_score']
#         print('flow_score', g_reverse.ndata['flow_score'].squeeze())
        
        

    def update_graph(self, g, eids, pre_pairs, post_pairs):
        # pre-compute queries and key_value pairs
        for pre_func, nids in pre_pairs:
            g.apply_nodes(pre_func, nids)
        self.propagate_attention(g, eids)
        # post-compute
        for post_func, nids in post_pairs:
            g.apply_nodes(post_func, nids)
            
    def initialize(self, queries, SG_queries):
        """
        queries: 1d tensor (batch_size,)
        SG_queries: list of tensor, each tensor is the index of neighbor events for queries
        """
        
        # query embedding
        sample_evts = np.array([[*self.id2evts[evt_idx][:-1]]for evt_idx in queries.numpy()])
        sample_sub = sample_evts[:, 0]
        sample_rel = sample_evts[:, 1]
        sample_ts = sample_evts[:, 3]

        sample_obj = sample_evts[:, 2]

        sample_sub_embed_th = self.event_encoder.ent_raw_embed(
            torch.from_numpy(sample_sub).to(torch.int64).to(self.device)).to(torch.float32)
        sample_rel_embed_th = self.event_encoder.rel_raw_embed(
            torch.from_numpy(sample_rel).to(torch.int64).to(self.device)).to(torch.float32)
        sample_ts_embed_th = self.event_encoder.ts_raw_embed(
            torch.from_numpy(sample_ts[:, np.newaxis]).to(torch.int64).to(self.device)).squeeze(1).to(torch.float32)
        sample_embed_cat = torch.cat([sample_sub_embed_th, sample_rel_embed_th, sample_ts_embed_th], axis=1)
        
#         SG_list = get_independent_subgraphs(self.G, SG_queries)
        SG_list = self.G.subgraphs(SG_queries)
        self.sample_query_embed = sample_embed_cat
#         self.entity_flow_score = torch.zeros(len(queries), num_entity)
#         self.entity_flow_score[torch.arange(len(queries)).to(torch.int64), torch.from_numpy(sample_sub).to(torch.int64)] = 1
        
        
        for i, sg in enumerate(SG_list):
            new_sg = dgl.transform.add_self_loop(sg)
            for k, v in sg.ndata.items():
                new_sg.ndata[k] = v
            sg = new_sg
            # query embedding
            context = self.sample_query_embed[i].expand(sg.number_of_nodes(), -1) # expand create a new view rather than allocating new memory
            sg.ndata['query_embed'] = context

            # event embedding
            sg_evts = np.array([[*self.id2evts[evt_idx][:-1]] for evt_idx in sg.ndata['_ID'].numpy()])
            sg_event_embed = self.event_encoder(sg_evts)
            sg.ndata['event_embed'] = sg_event_embed

            # init score
            init_flow_score = torch.ones(sg.number_of_nodes(),self.h,1, requires_grad=False)/sg.number_of_nodes()
            sg.ndata['flow_score'] = init_flow_score.to(self.device)
            SG_list[i] = sg
            
        bg = dgl.batch(SG_list)
    
        pre_func = self.encoder.pre_func(0, 'qkv')
        post_func = self.encoder.post_func(0)
        self.update_graph(bg, bg.edges(), [(pre_func, bg.edges()[0])], [(post_func, bg.edges()[1])])

        # update entity_flow_score 
        SG_list = dgl.unbatch(bg)
        sparse_idx = np.array([[i, self.id2evts[evt][2]] for i, sg in enumerate(SG_list) for evt in sg.ndata['_ID'].numpy()])
        sparse_idx_th = torch.from_numpy(np.transpose(sparse_idx)).to(torch.int64).to(self.device)
        sparse_val_th = torch.cat([torch.mean(sg.ndata['flow_score'], dim=-2).view(-1) for sg in SG_list])
        # init flow score for sub
        sparse_idx_sub = np.vstack([np.arange(len(sample_sub))[np.newaxis, :], sample_sub[np.newaxis, :]])
        sparse_idx_sub_th = torch.from_numpy(sparse_idx_sub).to(torch.int64).to(self.device)
        sparse_idx_th = torch.cat([sparse_idx_th, sparse_idx_sub_th], dim=1)
        sparse_val_th = torch.cat([sparse_val_th, torch.ones(len(sample_sub)).to(self.device)])
        
        self.entity_flow_score = torch.sparse.FloatTensor(sparse_idx_th, sparse_val_th, torch.Size([len(SG_list), self.num_entity])).to_dense()/2
#         print(self.entity_flow_score[0][torch.nonzero(self.entity_flow_score[0]).squeeze()])
        
        # print init events and flow score
#         for sg_idx, sg in enumerate(SG_list):
#             print("events in initialized subgraph {}: ".format(sg_idx))
#             for idx, evt in enumerate(sg.ndata['_ID'].numpy()):
#                 quad = self.id2evts[evt]
#                 print("{} : {}, {}, {}, {}: {}".format(
#                     quad,
#                     contents.id2entity[quad[0]], 
#                     contents.id2relation[quad[1]], 
#                     contents.id2entity[quad[2]], 
#                     quad[3], 
#                     sg.nodes[idx].data['flow_score'].item()))
#             print("init entity flow score:")
#             nonzero_ent = torch.nonzero(self.entity_flow_score[sg_idx], as_tuple=True)[0]
#             nonzero_ent_score = self.entity_flow_score[sg_idx][nonzero_ent]
#             for ent, score in zip(nonzero_ent.detach().numpy(), nonzero_ent_score.detach().numpy()):
#                 print("{}({}): {}".format(ent, contents.id2entity[ent], score))
        return dgl.unbatch(bg)
        
    
    def entity_score_update(self, edge_score, edges):
        """
        return normalized transition matrix on entity level
        edge_score: 2d tensor (num_edges, )
        edges: 2d numpy array (num_edges, 2)
        num_entity: int
        """
#         selfloop_u = selfloop_v = [_ for _ in range(num_entity)]

        entities_in = np.unique(edges)
        entities_out = np.setdiff1d(np.arange(self.num_entity), entities_in)
        edge_u, edge_v = np.transpose(edges)
        sparse_idx = np.vstack([np.concatenate([edge_v, edge_u]), np.concatenate([edge_u, edge_v])])
        sparse_idx_th = torch.from_numpy(sparse_idx).to(self.device)
        sub_segment_id = sparse_idx[1]
        
        val = edge_score.repeat(2)
        xv_sub, yv_sub = np.meshgrid(sub_segment_id, sub_segment_id)
        sparse_val_sub_norm_den = torch.mm(
            torch.from_numpy(xv_sub==yv_sub).to(torch.float32).to(self.device), 
            val.view(-1,1)).view(-1)
        # xv_obj, yv_obj = np.meshgrid(obj_segment_id, obj_segment_id)
        # sparse_val_obj_norm_den = torch.mm(torch.from_numpy(xv_obj==yv_obj).to(torch.float32), val.view(-1,1)).view(-1)

        sparse_val_sub2obj_th = torch.div(val, sparse_val_sub_norm_den).to(torch.float32)
        # sparse_val_obj2sub_th = torch.div(val, sparse_val_obj_norm_den).to(torch.float32)

        sparse_idx_th = torch.cat([
            sparse_idx_th, 
            torch.from_numpy(entities_in).repeat(2, 1).to(self.device), 
            torch.from_numpy(entities_out).repeat(2,1).to(self.device)], dim=1)
        sparse_val_th = torch.cat([0.8*sparse_val_sub2obj_th, 
                                   0.2*torch.ones(len(entities_in)).to(self.device), 
                                   torch.ones(len(entities_out)).to(self.device)])
#         print(sparse_idx_th.shape, sparse_val_th.shape)
        tran = torch.sparse.FloatTensor(
            sparse_idx_th, sparse_val_th, 
            torch.Size([self.num_entity, self.num_entity]))
        return tran
    
    
    def forward(self, SG_nodes_l, SG_event_embed, SG_query_embed, SG_flow_score):
        """
        SG_nodes_l: list of 1d tensor
        SG_event_embed: list of 2d tensor
        SG_query_embed: list of 2d tensor
        SG_flow_score: list of 2d tensor (num_nodes, num_heads)
        """
        for i in range(1, self.encoder.N):
            # dynamically expand graph
            # flow_score initialization against edge direction
            t0 = time.time()
            new_SG_list = []
            for group_idx, sg_nodes in enumerate(SG_nodes_l):
                neighbor = torch.cat([sub.layer_parent_nid(0)[-self.num_neighbors:] 
                                      for sub in dgl.contrib.sampling.sampler.NeighborSampler(
                                          G, 1, expand_factor=10000, num_hops=1, seed_nodes = sg_nodes, 
                                          add_self_loop=False)])
                sub_nodes = torch.unique(torch.cat([neighbor, sg_nodes]))
#                 sg_exp = get_independent_subgraph(self.G, sub_nodes)
                sg_exp = self.G.subgraph(sub_nodes)
                
                sg_evts = np.array([[*self.id2evts[evt_idx][:-1]] for evt_idx in sub_nodes.numpy()])
                # init node embedding
                sg_exp.ndata['event_embed'] = self.event_encoder(sg_evts)
                sg_exp.ndata['query_embed'] = self.sample_query_embed[group_idx].expand(sg_exp.number_of_nodes(), -1)
                # update node embedding and flow score from previous subgraph
                sg_exp.nodes[sg_exp.map_to_subgraph_nid(sg_nodes)].data['event_embed'] = SG_event_embed[group_idx]
                sg_exp.nodes[sg_exp.map_to_subgraph_nid(sg_nodes)].data['flow_score'] = SG_flow_score[group_idx]
                
                # add selfloop
                new_sg_exp = dgl.transform.add_self_loop(sg_exp)
                for k, v in sg_exp.ndata.items():
                    new_sg_exp.ndata[k] = v
                sg_exp = new_sg_exp
                
                # init node flow score for new neighbors
                # this step is a naive pagerank
                # note that this implementation also spread flow score between nodes in the previous subgraph
                # due to implementation difficulity
                sg_exp_reverse = dgl.transform.reverse(sg_exp, share_ndata=True)
                sg_exp_reverse.ndata['deg'] = sg_exp_reverse.out_degrees().view(-1,1,1).to(self.device)
                sg_exp_reverse.ndata['pv_sum'] = torch.zeros(sg_exp_reverse.number_of_nodes(),1,1).to(self.device)
                sg_exp_reverse.ndata['pv'] = sg_exp_reverse.ndata['flow_score']/sg_exp_reverse.ndata['deg']
                
                sg_exp_reverse.update_all(message_func=fn.copy_u('pv','pv'),reduce_func=fn.sum('pv', 'pv_sum'))
                sg_exp_reverse.apply_nodes(lambda x: {'flow_score': x.data['pv_sum']+x.data['flow_score']})
                sg_exp.ndata['flow_score'] = sg_exp_reverse.ndata['flow_score']
                
                
                new_SG_list.append(sg_exp)
#             for sg_idx, sg in enumerate(new_SG_list):
#                 print("events in expanded subgraph {}: ".format(sg_idx))
#                 for idx, evt in enumerate(sg.ndata['_ID'].numpy()):
#                     quad = self.id2evts[evt]
#                     print(quad, ": {}, {}, {}, {}: {}".format(
#                         contents.id2entity[quad[0]], 
#                         contents.id2relation[quad[1]], 
#                         contents.id2entity[quad[2]], 
#                         quad[3], 
#                         sg.nodes[idx].data['flow_score'].item()))
                
                    
                    
                
            t_expand = time.time()
#             print('expansion:', t_expand-t0)
            # update flow score, and node embedding
            bg = dgl.batch(new_SG_list, node_attrs=['event_embed', 'query_embed', 'flow_score', '_ID'])
            pre_func = self.encoder.pre_func(i, 'qkv')
            post_func = self.encoder.post_func(i)
            edges = bg.edges()
            self.update_graph(bg, edges, [(pre_func, edges[0])], [(post_func, edges[1])])
            
                
            
            t_update = time.time()
#             print('update:', t_update-t_expand)
            # pruning
            SG_list = dgl.unbatch(bg)
            pruned_SG_nodes_indices = []
            pruned_SG_nodes_flow_score = []
            pruned_SG_nodes_event_embed = []
            pruned_SG_nodes_query_embed = []
#             pruned_SG_list = []
            entity_score_transition_matrix = []
            for sg_idx, sg in enumerate(SG_list):
                if self.max_sg_num_nodes < sg.number_of_nodes():
                    _, topk_indices = torch.topk(torch.mean(sg.ndata['flow_score'], dim=-2).view(-1), self.max_sg_num_nodes)
                    topk_parent_nid = sg.ndata['_ID'][topk_indices]
                    pruned_SG_nodes_indices.append(topk_parent_nid)
                    flow_score_raw = sg.nodes[topk_indices].data['flow_score']
                    flow_score_norm = torch.div(flow_score_raw, torch.sum(flow_score_raw, dim=0, keepdim=True))
                    pruned_SG_nodes_flow_score.append(flow_score_norm)
                    pruned_SG_nodes_event_embed.append(sg.nodes[topk_indices].data['event_embed'])
                    pruned_SG_nodes_query_embed.append(sg.nodes[topk_indices].data['query_embed'])
                    
#                     sg_pruned = self.G.subgraph(topk_parent_nid)
#                     sg_pruned.ndata['flow_score'] = sg.nodes[topk_indices].data['flow_score']
#                     # TBD: normalization of flow_score
#                     sg_pruned.ndata['event_embed'] = sg.nodes[topk_indices].data['event_embed']
#                     sg_pruned.ndata['query_embed'] = sg.nodes[topk_indices].data['query_embed']
#                     SG_list[sg_idx] = sg_pruned
                else:
                    pruned_SG_nodes_indices.append(sg.ndata['_ID'])
                    pruned_SG_nodes_flow_score.append(sg.ndata['flow_score'])
                    pruned_SG_nodes_event_embed.append(sg.ndata['event_embed'])
                    pruned_SG_nodes_query_embed.append(sg.ndata['query_embed'])
#                     topk_parent_nid = sg.ndata['_ID']
#                     topk_indices = torch.arange(number_of_nodes())
                    
#                     sg_pruned = self.G.subgraph(topk_parent_nid)
#                     sg_pruned.ndata['flow_score'] = sg.nodes[topk_indices].data['flow_score']
#                     # TBD: normalization of flow_score
#                     sg_pruned.ndata['event_embed'] = sg.nodes[topk_indices].data['event_embed']
#                     sg_pruned.ndata['query_embed'] = sg.nodes[topk_indices].data['query_embed']

#                     SG_list[sg_idx] = sg
    
#                 print("events in pruned subgraph {}: ".format(sg_idx))
#                 for idx, evt in enumerate(pruned_SG_nodes_indices[sg_idx].numpy()):
#                     quad = self.id2evts[evt]
#                     print(quad, ": {}, {}, {}, {}: {}".format(
#                         contents.id2entity[quad[0]], 
#                         contents.id2relation[quad[1]], 
#                         contents.id2entity[quad[2]], 
#                         quad[3], 
#                         pruned_SG_nodes_flow_score[sg_idx][idx].item()
#                     ))
                    
                edges = np.array([[self.id2evts[evt][2], self.id2evts[evt][0]] 
                                  for evt in pruned_SG_nodes_indices[sg_idx].numpy()])
                entity_score_transition_matrix.append(
                    self.entity_score_update(
                        torch.mean(pruned_SG_nodes_flow_score[sg_idx], dim=-2).view(-1), 
                        edges))
                
                    
#                 sparse_idx_sub2obj = np.array([[self.id2evts[evt][2], self.id2evts[evt][0]] for evt in pruned_SG_nodes_indices[sg_idx].numpy()])
#                 sparse_idx_obj2sub = np.array([[self.id2evts[evt][0], self.id2evts[evt][2]] for evt in pruned_SG_nodes_indices[sg_idx].numpy()])
#                 sparse_idx = np.concatenate([sparse_idx_sub2obj, sparse_idx_obj2sub], axis=0)
                
#                 sparse_idx_th = torch.from_numpy(np.transpose(sparse_idx))
                
#                 sparse_val_th = pruned_SG_nodes_flow_score[sg_idx].squeeze().repeat(2)
#                 #normalize transition matrix on entity level
#                 sub_segment_id = np.transpose(sparse_idx_obj2sub).reshape(-1) # subject of all edges, including reverse edge
                
#                 xv_sub, yv_sub = np.meshgrid(sub_segment_id, sub_segment_id)
#                 sparse_val_sub_norm_den = torch.mm(torch.from_numpy(xv_sub==yv_sub).to(torch.float32), sparse_val_th.view(-1,1)).view(-1)
                
#                 sparse_val_th = torch.div(sparse_val_th, sparse_val_sub_norm_den).to(torch.float32)
                
                
#                 # add self loop
#                 sparse_idx_th = torch.cat([sparse_idx_th, torch.arange(self.num_entity).repeat(2, 1)], dim=1)
#                 sparse_val_th = torch.cat([0.8*sparse_val_th, 0.2*torch.ones(self.num_entity)])
#                 entity_score_transition_matrix.append(torch.sparse.FloatTensor(
#                     sparse_idx_th, sparse_val_th, 
#                     torch.Size([self.num_entity,self.num_entity])))
               
            t_prun = time.time()
#             print('pruning:', t_prun-t_update)
            
            # update entity score
            updated_entity_flow_score = torch.cat([
                torch.sparse.mm(trans_sp, self.entity_flow_score[i, :].view(-1,1)).view(1,-1) 
                for i, trans_sp in enumerate(entity_score_transition_matrix)])
            self.entity_flow_score = updated_entity_flow_score
             
#             for sg_idx, sg in enumerate(SG_list):
#                 print("entities with non-zero flow score in subgraph {}: ".format(sg_idx))
#                 nonzero_ent = torch.nonzero(self.entity_flow_score[sg_idx], as_tuple=True)[0]
#                 nonzero_ent_score = self.entity_flow_score[sg_idx][nonzero_ent]
#                 for ent, score in zip(nonzero_ent.detach().numpy(), nonzero_ent_score.detach().numpy()):
#                     print("{}({}): {}".format(ent, contents.id2entity[ent], score))
            
            t_ent = time.time()
#             print('update entity:', t_ent-t_prun)
#             print(self.entity_flow_score[0][torch.nonzero(self.entity_flow_score[0]).squeeze()])
            SG_nodes_l = pruned_SG_nodes_indices
            SG_event_embed = pruned_SG_nodes_event_embed
            SG_query_embed = pruned_SG_nodes_query_embed
            SG_flow_score = pruned_SG_nodes_flow_score

    
class TimeEncode(torch.nn.Module):
    '''
    This class implemented the Bochner's time embedding
    '''

    def __init__(self, dim_model, device='cpu'):
        '''

        :param expand_dim: number of samples draw from p(w), which are used to estimate kernel based on MCMC
        refer to Self-attention with Functional Time Representation Learning for more detail
        '''
        super(TimeEncode, self).__init__()

        time_dim = dim_model
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
    
class EventEncode(torch.nn.Module):
    def __init__(self, dim_model, id2entity, id2relation):
        super(EventEncode, self).__init__()
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.ent_raw_embed = torch.nn.Embedding(len(id2entity), dim_model)
        self.rel_raw_embed = torch.nn.Embedding(len(id2relation), dim_model)
        self.ts_raw_embed = TimeEncode(dim_model)
        
    def forward(self, sg_evts):
        """
        events_idx: 1d tensor of events
        """
        device = next(self.ent_raw_embed.parameters()).device
        sg_sub = sg_evts[:, 0]
        sg_rel = sg_evts[:, 1]
        sg_obj = sg_evts[:, 2]
        sg_ts = sg_evts[:, 3]
        sg_sub_embed_th = self.ent_raw_embed(torch.from_numpy(sg_sub).to(torch.int64).to(device))
        sg_rel_embed_th = self.rel_raw_embed(torch.from_numpy(sg_rel).to(torch.int64).to(device))
        sg_obj_embed_th = self.ent_raw_embed(torch.from_numpy(sg_obj).to(torch.int64).to(device))
        sg_ts_embed_th = self.ts_raw_embed(torch.from_numpy(sg_ts[:, np.newaxis]).to(torch.int64).to(device)).squeeze(1)
        sg_embed_cat = torch.cat([sg_sub_embed_th, sg_rel_embed_th, sg_obj_embed_th, sg_ts_embed_th], axis=1)
        return sg_embed_cat
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ICEWS14_forecasting', help='specify data set')
parser.add_argument('--dim_model', type=int, default=32, help='dimension of embedding for node, realtion and time')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--device', type=int, default=-1, help='-1: cpu, >=0, cuda device')
parser.add_argument('--DP_steps', type=int, default=3, help='number of DP steps')
parser.add_argument('--head', type=int, default=2, help='number of head')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout for encoder')
parser.add_argument('--num_neighbors', type=int, default=10, help='number of neighbors sampled for sampling horizon')
parser.add_argument('--max_attended_nodes', type=int, default=20, help='max number of nodes in attending from horizon')
parser.add_argument('--load_checkpoint', type=str, default=None, help='train from checkpoints')
parser.add_argument('--debug', action='store_true', default=None, help='in debug mode, checkpoint will not be saved')
args = parser.parse_args()


if __name__ == '__main__':
    print(args)

    # check cuda
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.device) if args.device >= 0 else 'cpu'
    else:
        device = 'cpu'
        
    contents = Data(dataset=args.dataset, add_reverse_relation=False)
    train_valid_data = np.concatenate([contents.train_data, contents.valid_data_seen_entity], axis=0)
    sp2o = contents.get_sp2o()
    
    id2evts = {k:tuple(v) for k, v in enumerate(train_valid_data)}
    sub2evt = defaultdict(list)
    obj2evt = defaultdict(list)

    for i, evt in enumerate(train_valid_data):
        sub2evt[evt[0]].append(i)
        obj2evt[evt[2]].append(i)
        
    # filter out event with subject appearing for the first time
    sub_exists = [False]*len(contents.id2entity)
    train_fil = []

    for i, evt in enumerate(contents.train_data):
        if not sub_exists[evt[0]]:
            for ngh in sub2evt[evt[0]]:
                if id2evts[ngh][3] < evt[3]:
                    sub_exists[evt[0]] = True
                    break
        else:
            train_fil.append(i)
            
    valid_fil = [i for i in range(len(contents.train_data), len(train_valid_data))]
            
    # construct Event Graph
    sub_sub_edges = []
    sub_sub_edges_weight = []
    for gr in sub2evt.values():
        for i, j in itertools.combinations(gr, r=2):
            if train_valid_data[j, 3] - train_valid_data[i,3] >= 0:
                sub_sub_edges.append((i, j))
                sub_sub_edges_weight.append(train_valid_data[j,3]-train_valid_data[i,3])

    obj_obj_edges = []
    obj_obj_edges_weight = []
    for gr in obj2evt.values():
        for i, j in itertools.combinations(gr, r=2):
            if train_valid_data[j, 3] - train_valid_data[i, 3] >= 0:
                obj_obj_edges.append((i,j))
                obj_obj_edges_weight.append(train_valid_data[j, 3] - train_valid_data[i, 3])

    sub_obj_edges = []
    sub_obj_edges_weight = []
    for sub, sub_evt in sub2evt.items():
        for i, j in itertools.product(sub_evt, obj2evt[sub]):
            if train_valid_data[j,3] > train_valid_data[i,3]: # this relation doesn't exist when two events happen simultaneously
                sub_obj_edges.append((i, j))
                sub_obj_edges_weight.append(train_valid_data[j,3]-train_valid_data[i,3])

    obj_sub_edges = []
    obj_sub_edges_weight = []
    for obj, obj_evt in obj2evt.items():
        for i, j in itertools.product(obj_evt, sub2evt[obj]):
            if train_valid_data[j,3] > train_valid_data[i,3]: # this relation doesn't exist when two events happen simultaneously
                obj_sub_edges.append((i, j))
                obj_sub_edges_weight.append(train_valid_data[j,3]-train_valid_data[i,3])
                
    # edge point from event happen earlier to later
    G = dgl.DGLGraph()
    G.add_nodes(len(train_valid_data)+1)
    edges = sub_sub_edges + obj_obj_edges + sub_obj_edges + obj_sub_edges
    temp = np.array(edges)
    u,v = temp[:, 0], temp[:, 1]
    G.add_edges(u,v)
    G.edata['weight'] = torch.tensor(sub_sub_edges_weight+obj_obj_edges_weight+sub_obj_edges_weight+obj_sub_edges_weight)
    G.readonly()
    
    del sub_sub_edges
    del sub_sub_edges_weight
    del obj_obj_edges
    del obj_obj_edges_weight
    del obj_sub_edges
    del obj_sub_edges_weight
    del sub_obj_edges
    del sub_obj_edges_weight
    
    # note that there is difference between current model and TGAN: no offset for entity and relation identity in embedding
    num_entity = len(contents.id2entity)
    num_relation = len(contents.id2relation)
    
    attn = MultiHeadGATLayer(args.dim_model, args.head)
    ff = torch.nn.Linear(4*args.dim_model, 4*args.dim_model) # feed forward after multihead concatenatation
    encoder = Encoder(EncoderLayer(args.dim_model, copy.deepcopy(attn), copy.deepcopy(ff), args.dropout), args.DP_steps) # event node embedding
    event_encoder = EventEncode(args.dim_model, contents.id2entity, contents.id2relation)
    model = tE2GN(G, encoder, args.dim_model, args.head, event_encoder, id2evts, num_entity, num_relation,
            num_neighbors=args.num_neighbors, max_sg_num_nodes=args.max_attended_nodes, device=device)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, ":", param.shape, param.device)
    
    # save configuration
    start_time = time.time()
    struct_time = time.gmtime(start_time)
    
    start_epoch = 0
    
    if not args.debug:
        if args.load_checkpoint is None:
            CHECKPOINT_PATH = os.path.join(save_dir, 'Checkpoints', 'tE2Net', 'checkpoints_{}_{}_{}_{}_{}_{}'.format(
                struct_time.tm_year,
                struct_time.tm_mon,
                struct_time.tm_mday,
                struct_time.tm_hour,
                struct_time.tm_min,
                struct_time.tm_sec))
            if not os.path.exists(CHECKPOINT_PATH):
                os.makedirs(CHECKPOINT_PATH, mode=0o770)
        else:
            CHECKPOINT_PATH = os.path.join(save_dir, 'Checkpoints', 'tE2Net', os.path.dirname(args.load_checkpoint))
        print("Save checkpoints under {}".format(CHECKPOINT_PATH))
        save_config(args, CHECKPOINT_PATH)
        print("Log configuration under {}".format(CHECKPOINT_PATH))
        
    if args.load_checkpoint:
        model, optimizer, start_epoch = load_checkpoint(
            model, optimizer, 
            os.path.join(save_dir, 'Checkpoints', 'tE2Net', args.load_checkpoint))
        start_epoch += 1
        print("Load checkpoints from {}".format(args.checkpoint))
    
    for epoch in range(start_epoch, args.epoch):
        
        running_loss = 0.
        for batch_idx, sample in tqdm(enumerate(DataLoader(train_fil, batch_size=args.batch_size, shuffle=True))):
            optimizer.zero_grad()
            model.zero_grad()
            model.train()

            SG_queries = [[ngh for ngh in sub2evt[id2evts[query][0]] 
                       if id2evts[ngh][3] < id2evts[query][3]][-args.num_neighbors:]
                      for query in sample.numpy()]
            SG_list = model.initialize(sample, SG_queries)
            SG_nodes_l = [sg.ndata['_ID'] for sg in SG_list]
            SG_event_embed = [sg.ndata['event_embed'] for sg in SG_list] 
            SG_query_embed = [sg.ndata['query_embed'] for sg in SG_list] 
            SG_flow_score = [sg.ndata['flow_score'] for sg in SG_list]
        #     print(SG_list[0].number_of_nodes())
            model(SG_nodes_l, SG_event_embed, SG_query_embed, SG_flow_score)
            # logits = torch.nn.Softmax(dim=1)(model.entity_flow_score+1e-20)
            logits = torch.div(model.entity_flow_score+1e-20, torch.sum(model.entity_flow_score+1e-20, dim=-1, keepdim=True))
            one_hot_label = torch.tensor(np.array([np.arange(num_entity) == id2evts[evt][2] for evt in sample.numpy()], dtype=np.float32)).to(device)
            loss = torch.mean(torch.sum(torch.nn.BCELoss(reduction='none')(logits, one_hot_label), dim=-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 10 == 9:
                print('[%d, %5d] training loss: %.3f' % (epoch, batch_idx, running_loss / 10))
                running_loss = 0.0
                
        model.eval()
        if not args.debug:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ent_raw_embed_state_dict': ent_raw_embed.state_dict(),
                'rel_raw_embed_state_dict': rel_raw_embed.state_dict(),
                'ts_raw_embed_state_dict': ts_raw_embed.state_dict(),
                'loss': loss,
            }, os.path.join(CHECKPOINT_PATH, 'checkpoint_{}.pt'.format(epoch)))
        if epoch % 1 == 0:
            hit_1 = hit_3 = hit_10 = 0
            mr = mrr =0
            num_query = 0
            
            for batch_idx, sample in enumerate(DataLoader(
                valid_filter, batch_size=args.batch_size, shuffle=True)):
                
                SG_queries = [[ngh for ngh in sub2evt[id2evts[query][0]] 
                       if id2evts[ngh][3] < id2evts[query][3]][-args.num_neighbors:]
                      for query in sample.numpy()]
                num_query += len(SG_query)
                SG_list = model.initialize(sample, SG_queries)
                SG_nodes_l = [sg.ndata['_ID'] for sg in SG_list]
                SG_event_embed = [sg.ndata['event_embed'] for sg in SG_list] 
                SG_query_embed = [sg.ndata['query_embed'] for sg in SG_list] 
                SG_flow_score = [sg.ndata['flow_score'] for sg in SG_list]
                model(SG_nodes_l, SG_event_embed, SG_query_embed, SG_flow_score)
                sample_np = sample.numpy()
                hit_1_batch, hit_3_batch, hit_10_batch, mr_batch, mrr_batch = calc_metric(sample_np[:, 0], sample_np[:, 1], model.entity_flow_score.detach().numpy(), sample_np[:, 2], sp2o, num_entity)
                hit_1 += hit_1_batch
                hit_3 += hit_3_batch
                hit_10 += hit_10_batch
                mr += mr_batch
                mrr += mrr_batch
                
            print(
            "Filtered performance: Hits@1: {}, Hits@3: {}, Hits@10: {}, MR: {}, MRR: {}".format(
                hit_1/num_query,
                hit_3/num_query,
                hit_10/num_query,
                mr/num_query,
                mrr/num_query))
                 
    print("finished Training")

    
    
