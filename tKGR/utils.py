import os

import numpy as np

import tKGR.data
DataDir = os.path.dirname(tKGR.data.__file__)

class Data:
    def __init__(self, dataset=None):
        # load data
        self.train_data = self._load_data(os.path.join(DataDir, dataset), "train")
        self.valid_data = self._load_data(os.path.join(DataDir, dataset), "valid")
        self.test_data = self._load_data(os.path.join(DataDir, dataset), "test")
        self.id2entity = self._id2entity(dataset=dataset)
        self.id2relation = self._id2relation(dataset=dataset)
        self.data = self.train_data + self.valid_data + self.test_data


        self.entities = self._get_entities(self.data)
        self.train_relations = self._get_relations(self.train_data)
        self.valid_relations = self._get_relations(self.valid_data)
        self.test_relations = self._get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations
                                                 if i not in self.train_relations] + [i for i in self.test_relations
                                                                                      if i not in self.train_relations]
        self.timestamps = self._get_timestamps(self.data)

        self.entity_idxs, self.relation_idxs, self.timestamp_idxs = self._get_idx()

    def _load_data(self, data_dir, data_type="train"):
        with open(os.path.join(data_dir, "{}.txt".format(data_type))) as f:
            data = f.readlines()
            data = [line.split("\t") for line in data] #only cut by "\t", not by white space.
            data = [[_.strip() for _ in line] for line in data] # remove white space
            data += [[i[2], i[1]+"_reverse", i[0], i[3]] for i in data]
        return data

    def _get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def _get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

    def _get_timestamps(self, data):
        timestamps = sorted(list(set([d[3] for d in data])))
        return timestamps

    def _get_idx(self):
        entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        relation_idxs = {self.relations[i]: i for i in range(len(self.relations))}
        timestamp_idxs = {self.timestamps[i]: i for i in range(len(self.timestamps))}
        return  entity_idxs, relation_idxs, timestamp_idxs
    
    def _id2entity(self, dataset):
        with open(os.path.join(DataDir, dataset, "entity2id.txt")) as f:
            mapping = f.readlines()
            mapping = [entity.strip().split("\t") for entity in mapping]
            mapping = {int(ent2idx[1].strip()):ent2idx[0].strip() for ent2idx in mapping}
        return mapping
    
    def _id2relation(self, dataset):
        with open(os.path.join(DataDir, dataset, "relation2id.txt")) as f:
            mapping = f.readlines()
            mapping = [relation.strip().split("\t") for relation in mapping]
            id2relation={}
            for rel2idx in mapping: 
                id2relation[rel2idx[1].strip()] = rel2idx[0].strip()
                id2relation[rel2idx[1].strip()+'_reverse'] = 'REVERSED ' + rel2idx[0].strip()
        return id2relation

class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """

        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l

        self.off_set_l = off_set_l

        self.uniform = uniform

    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]

        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[1])
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])

            off_set_l.append(len(n_idx_l))
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, off_set_l

    def find_before(self, src_idx, cut_time):
        """

        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l

        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_ts, neighbors_e_idx

        left = 0
        right = len(neighbors_idx) - 1

        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid

        if neighbors_ts[right] < cut_time:
            return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right]
        else:
            return neighbors_idx[:left], neighbors_e_idx[:left], neighbors_ts[:left]

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(src_idx_l) == len(cut_time_l))

        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)

            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)

                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]

                    assert (len(ngh_idx) <= num_neighbors)
                    assert (len(ngh_ts) <= num_neighbors)
                    assert (len(ngh_eidx) <= num_neighbors)

                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx):] = ngh_eidx

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch