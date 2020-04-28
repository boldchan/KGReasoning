import time
from collections import defaultdict
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import torch

from utils import NeighborFinder

class GraphGroup(NeighborFinder):
    def __init__(self, adj, sampling, max_time, num_entities, num_rel):
        super(GraphGroup, self).__init__(adj, sampling, max_time, num_entities)
        self.selfloop = num_rel

    def set_init(self, query_src_idx_l, query_rel_idx_l, query_tar_idx_l, query_cut_time_l):
        self.query_src_idx_l = query_src_idx_l
        self.query_rel_idx_l = query_rel_idx_l
        self.query_tar_idx_l = query_tar_idx_l
        self.query_cut_time_l = query_cut_time_l
        self.DP_step = 0
        self.num_query = len(query_src_idx_l)

        self.attending_nodes = np.stack([np.arange(self.num_query)[:, np.newaxis],
                                         self.query_src_idx_l[:, np.newaxis],
                                         self.query_cut_time_l[:, np.newaxis]])
        self.attending_nodes_score = np.ones(self.num_query)
        self.selected_edges = None

        self.GraphSlices = defaultdict(dict)
        self.GraphSlices[-1] = {'nodes': self.attending_nodes, 'score': self.attending_nodes_score, 'edge': self.selected_edges}

    def get_sampled_edges(self, attended_nodes, node_attention, num_neighbors, DP_step, tc=None):
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
        if tc:
            t_start = time.time()
        src_idx_l = attended_nodes[:, 1]
        cut_time_l = attended_nodes[:, 2]
        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.get_temporal_neighbor(
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

        self.GraphSlices[DP_step]['sampled_edges'] = sampled_edges
        if tc:
            tc['graph']['sample'] += time.time() - t_start
        return sampled_edges, src_attention

    def get_selected_edges(self, sampled_edges, tc=None):
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

    def topk_att_score(self, attending_nodes, attending_node_attention, k: int, DP_step, tc=None):
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

        attended_nodes = np.concatenate(res_nodes, axis=0)
        if tc:
            tc['graph']['topk'] += time.time() - t_start

        self.GraphSlices[DP_step]['attended_nodes'] = attended_nodes

        return attended_nodes, torch.cat(res_att, dim=0)

    def update_score(self, new_node_score, DP_step):
        """

        :param new_node_score: 1d numpy array of node score, which is called node attention score in earlier version
        :param DP_step:
        :return:
        """
        self.GraphSlices[DP_step]['score'] = new_node_score

    def plot_graph_til_t(self, query_idx, DP_step):
        """
        TBD: change to plot_graph_at_t
        plot interactively subgraph expasion  til step step
        :param t:
        :return:
        """
        G = nx.MultiDiGraph()
        for st in range(DP_step):
            sampled_edges = self.GraphSlices[st]['sampled_edges']
            sampled_edges = sampled_edges[sampled_edges[:, 0]==query_idx]
            edges = [((sampled_edges[i, 1], sampled_edges[i, 2]),
                      (sampled_edges[i, 3], sampled_edges[i, 4]),
                      dict(rel=sampled_edges[i, 5],
                           step=st))
                     for i in range(len(sampled_edges))]
            G.add_edges_from(edges)

        pos = nx.spring_layout(G, k=3 / np.sqrt(len(G.nodes)))

        edge_sub = defaultdict(list)
        edge_obj = defaultdict(list)
        edge_x = defaultdict(list)
        edge_y = defaultdict(list)
        middle_x = defaultdict(list)
        middle_y = defaultdict(list)
        rel = defaultdict(list)
        step = defaultdict(list)

        for n, nbrsdict in G.adjacency():
            for nbr, keydict in nbrsdict.items():
                for key, eattr in keydict.items():
                    edge_sub[eattr['step']].append(n)
                    edge_obj[eattr['step']].append(nbr)
                    edge_shift = 0. * np.random.randn()  # distinguish multiedges between same two nodes
                    edge_x[eattr['step']].append(pos[n][0] + edge_shift)
                    edge_x[eattr['step']].append(pos[nbr][0] + edge_shift)
                    edge_y[eattr['step']].append(pos[n][1] + edge_shift)
                    edge_y[eattr['step']].append(pos[nbr][1] + edge_shift)
                    middle_x[eattr['step']].append((pos[n][0] + pos[nbr][0]) / 2 + 0.01 * np.random.randn())
                    middle_y[eattr['step']].append((pos[n][1] + pos[nbr][1]) / 2 + 0.01 * np.random.randn())
                    rel[eattr['step']].append(eattr['rel'])
                    step[eattr['step']].append(eattr['step'])

        edge_color = ['rgb(179,179,179)', 'rgb(229,196,148)', 'rgb(255,217,47)', 'rgb(166,216,84)', 'rgb(231,138,195)']
        edge_trace = []
        middle_node_trace = []

        # TBD: handle self loop
        for st in range(DP_step):
            # https://plotly.com/python/hover-text-and-formatting/#adding-other-data-to-the-hover-with-customdata-and-a-hovertemplate
            edge_trace.append(go.Scatter(
                x=edge_x[st],
                y=edge_y[st],
                line=dict(
                    width=0.5,
                    color=edge_color[st]),
                mode='lines'
            ))

            middle_node_trace.append(go.Scatter(
                x=middle_x[st],
                y=middle_y[st],
                mode='markers',
                hovertemplate='%{text}',
                text=['sub: {}<br>obj: {}<br>rel: {}<br>step: {}'.format(s, o, r, st) for s, o, r, st in
                      zip(edge_sub[st], edge_obj[st], rel[st], step[st])],
                marker=go.Marker(
                    opacity=0
                )
            ))

        node_x = []
        node_y = []
        node_score = []
        for node in self.GraphSlices[DP_step]['node']:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_score.append(self.GraphSlices[DP_step]['score'])

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hovertemplate='%{text}',
            text = ["({}, {}): {}".format(node[0], node[1]) for node in G.nodes()],
            marker=dict(
                colorscale='YlGnBu',
                reversescale=True,
                color=list(G.degree()),
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )

        )

        fig = go.Figure(data=[node_trace, *edge_trace, *middle_node_trace],
                        layout=go.Layout(
                            title='<br>Subgraph expanded from ({}, {}, {}, {}) until {} step'.format(
                                self.query_src_idx_l[query_idx], self.query_rel_idx_l[query_idx], self.query_tar_idx_l[query_idx], self.query_cut_time_l[query_idx], DP_step),
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        fig.show()