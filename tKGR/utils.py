import os
import json

from collections import defaultdict
import numpy as np
import pdb

# import tKGR.data
DataDir = os.path.join(os.path.dirname(__file__), 'data')


class Data:
    def __init__(self, dataset=None):
        # load data
        self.id2entity = self._id2entity(dataset=dataset)
        self.id2relation = self._id2relation(dataset=dataset)
        num_relations = len(self.id2relation)  # number of pure relations, i.e. no reversed relation
        reversed_id2relation = {}
        for ind, rel in self.id2relation.items():
            reversed_id2relation[ind+num_relations] = 'Reversed '+rel
        self.id2relation.update(reversed_id2relation)

        self.num_relations = 2*num_relations
        self.num_entities = len(self.id2entity)

        self.train_data = self._load_data(os.path.join(DataDir, dataset), "train")
        self.valid_data = self._load_data(os.path.join(DataDir, dataset), "valid")
        self.test_data = self._load_data(os.path.join(DataDir, dataset), "test")

        # add reverse event into the data set
        self.train_data = np.concatenate([self.train_data[:, :-1],
                                          np.vstack(
                                              [[event[2], event[1]+num_relations, event[0], event[3]]
                                               for event in self.train_data])], axis=0)
        seen_entities = set(self.train_data[:, 0])
        seen_relations = set(self.train_data[:, 1])

        # remove events in valid data set and test data set that contains unseen entity and unseen relation
        val_mask = [evt[0] in seen_entities and evt[2] in seen_entities and evt[1] in seen_relations
                    for evt in self.valid_data]
        self.valid_data_seen_entity = self.valid_data[val_mask]
        self.valid_data = np.concatenate([self.valid_data[:, :-1],
                                          np.vstack(
                                              [[event[2], event[1]+num_relations, event[0], event[3]]
                                               for event in self.valid_data])], axis=0)
        self.valid_data_seen_entity = np.concatenate([self.valid_data_seen_entity[:, :-1],
                                          np.vstack(
                                              [[event[2], event[1]+num_relations, event[0], event[3]]
                                               for event in self.valid_data_seen_entity])], axis=0)

        test_mask = [evt[0] in seen_entities and evt[2] in seen_entities and evt[1] in seen_relations
                     for evt in self.test_data]
        self.test_data_seen_entity = self.test_data[test_mask]
        self.test_data = np.concatenate([self.test_data[:, :-1],
                                         np.vstack(
                                              [[event[2], event[1]+num_relations, event[0], event[3]]
                                               for event in self.test_data])], axis=0)
        self.test_data_seen_entity = np.concatenate([self.test_data_seen_entity[:, :-1],
                                          np.vstack(
                                              [[event[2], event[1]+num_relations, event[0], event[3]]
                                               for event in self.test_data_seen_entity])], axis=0)

        self.data = np.concatenate([self.train_data, self.valid_data, self.test_data], axis=0)

        # self.entities = _get_entities(self.data)
        # self.train_relations = _get_relations(self.train_data)
        # self.valid_relations = _get_relations(self.valid_data)
        # self.test_relations = _get_relations(self.test_data)
        # self.relations = self.train_relations + [i for i in self.valid_relations
        #                                          if i not in self.train_relations] + [i for i in self.test_relations
        #                                                                               if i not in self.train_relations]
        self.timestamps = self._get_timestamps(self.data)

        # self.entity_idxs, self.relation_idxs, self.timestamp_idxs = self._get_idx()

    def _load_data(self, data_dir, data_type="train"):
        with open(os.path.join(data_dir, "{}.txt".format(data_type)), 'r', encoding='utf-8') as f:
            data = f.readlines()
            data = np.array([line.split("\t") for line in data])  # only cut by "\t", not by white space.
            data = np.vstack([[int(_.strip()) for _ in line] for line in data])  # remove white space
        return data

    @staticmethod
    def _get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def _get_entities(data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

    @staticmethod
    def _get_timestamps(data):
        timestamps = np.array(sorted(list(set([d[3] for d in data]))))
        return timestamps

    def neg_sampling_object(self, Q, dataset='train', start_time=0):
        '''

        :param Q: number of negative sampling for each real quadruple
        :param start_time: neg sampling for events since start_time (inclusive), used for warm start training
        :param dataset: indicate which data set to choose negative sampling from
        :return:
        List[List[int]]: [len(train_data), Q], list of Q negative sampling for each event in train_data
        '''
        neg_object = []
        spt_o = defaultdict(list)  # dict: (s, p, r)--> [o]
        if dataset == 'train':
            contents_dataset = self.train_data
            assert start_time < max(self.train_data[:, 3])
        elif dataset == 'valid':
            contents_dataset = self.valid_data_seen_entity
            assert start_time < max(self.valid_data_seen_entity[:, 3])
        elif dataset == 'test':
            contents_dataset = self.test_data_seen_entity
            assert start_time < max(self.test_data_seen_entity)
        else:
            raise ValueError("invalid input for dataset, choose 'train', 'valid' or 'test'")

        data_after_start_time = [event for event in contents_dataset if event[3] >= start_time]
        for event in data_after_start_time:
            spt_o[(event[0], event[1], event[3])].append(event[2])
        for event in data_after_start_time:
            neg_object_one_node = []
            while True:
                candidate = np.random.choice(self.num_entities)
                if candidate not in spt_o[(event[0], event[1], event[3])]:
                    neg_object_one_node.append(candidate) # 0-th is a dummy node used to stuff the neighborhood when there is not enough nodes in the neighborhood
                if len(neg_object_one_node) == Q:
                    neg_object.append(neg_object_one_node)
                    break

        return np.stack(neg_object, axis=0)

    def _get_idx(self):
        entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        relation_idxs = {self.relations[i]: i for i in range(len(self.relations))}
        timestamp_idxs = {self.timestamps[i]: i for i in range(len(self.timestamps))}
        return  entity_idxs, relation_idxs, timestamp_idxs
    
    def _id2entity(self, dataset):
        with open(os.path.join(DataDir, dataset, "entity2id.txt"), 'r', encoding='utf-8') as f:
            mapping = f.readlines()
            mapping = [entity.strip().split("\t") for entity in mapping]
            mapping = {int(ent2idx[1].strip()):ent2idx[0].strip() for ent2idx in mapping}
        return mapping
    
    def _id2relation(self, dataset):
        with open(os.path.join(DataDir, dataset, "relation2id.txt"), 'r', encoding='utf-8') as f:
            mapping = f.readlines()
            mapping = [relation.strip().split("\t") for relation in mapping]
            id2relation = {}
            for rel2idx in mapping: 
                id2relation[int(rel2idx[1].strip())] = rel2idx[0].strip()
        return id2relation

    def get_adj_list(self):
        '''
        adj_list for the whole dataset, including training data, validation data and test data
        :return:
        adj_list: List[List[(o(int), p(str), t(int))]], adj_list[i] is the list of (o,p,t) of events where entity i is the subject
        '''
        adj_list_dict = defaultdict(list)
        for event in self.data:
            adj_list_dict[int(event[0])].append((int(event[2]), event[1], int(event[3])))

        subject_index_sorted = sorted(adj_list_dict.keys())
        adj_list = [sorted(adj_list_dict[_], key=lambda x: x[2]) for _ in subject_index_sorted]

        return adj_list

    def get_spt2o(self, dataset:str):
        '''
        mapping between (s, p, t) -> list(o), i.e. values of dict are objects share the same subject, predicate and time.
        calculated for the convenience of evaluation "fil" on object prediction
        :param dataset: 'train', 'valid', 'test'
        :return:
        dict (s, p, t) -> o
        '''
        if dataset == 'train':
            events = self.train_data
        elif dataset == 'valid':
            events = self.valid_data
        elif dataset == 'test':
            events = self.test_data
        else:
            raise ValueError("invalid input {} for dataset, please input 'train', 'valid' or 'test'".format(dataset))
        spt2o = defaultdict(list)
        for event in events:
            spt2o[(event[0], event[1], event[3])].append(event[2])
        return spt2o

    def get_sp2o(self):
        '''
        get dict d which mapping between (s, p) -> list(o). More specifically, for each event in the **whole data set**,
        including training, validation and test data set, its object will be in d[(s,p)]
        it's calculated for the convenience of a looser evaluation "fil" on object prediction
        :param dataset: 'train', 'valid', 'test'
        :return:
        dict (s, p) -> o
        '''
        sp2o = defaultdict(list)
        for event in self.data:
            sp2o[(event[0], event[1])].append(event[2])
        return sp2o

class NeighborFinder:
    def __init__(self, adj_list, uniform=False, max_time=366*24):
        """
        Params
        ------
        adj_list: adj_list[i] is the list of all (o,p,t) for entity i
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i][:,1]
        off_set_t_l: node_idx_l[off_set_l[i]:off_set_l[i + 1]][:off_set_t_l[i][cut_time/24]] --> object of entity i that happen before cut time
        """

        node_idx_l, node_ts_l, edge_idx_l, off_set_l, off_set_t_l = self.init_off_set(adj_list, max_time)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l

        self.off_set_l = off_set_l
        self.off_set_t_l = off_set_t_l

        self.uniform = uniform

    def init_off_set(self, adj_list, max_time):
        """
        Params
        ------
        adj_list: List[List[int]]

        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        off_set_t_l = []

        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: int(x[2]))
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            curr_ts = [x[2] for x in curr]
            n_ts_l.extend(curr_ts)

            off_set_l.append(len(n_idx_l))
            off_set_t_l.append([np.searchsorted(curr_ts, cut_time, 'left') for cut_time in range(0, max_time, 24)])
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, off_set_l, off_set_t_l

    def find_before(self, src_idx, cut_time):
        """
        build neighborhood sequence of entity sec_idx before cut_time
        Params
        ------
        src_idx: int
        cut_time: float
        """
        neighbors_idx = self.node_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        neighbors_ts = self.node_ts_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
        neighbors_e_idx = self.edge_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]

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
            return neighbors_idx[:right+1], neighbors_e_idx[:right+1], neighbors_ts[:right+1]
        else:
            return neighbors_idx[:left+1], neighbors_e_idx[:left+1], neighbors_ts[:left+1]

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        each entity has exact num_neighbors neighbors, either by uniform sampling or by picking from the first
        num_neighbors events
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
            neighbors_idx = self.node_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            neighbors_ts = self.node_ts_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            neighbors_e_idx = self.edge_idx_l[self.off_set_l[src_idx]:self.off_set_l[src_idx + 1]]
            mid = self.off_set_t_l[src_idx][int(cut_time/24)]
            # mid = np.searchsorted(neighbors_ts, cut_time)
            ngh_idx, ngh_eidx, ngh_ts = neighbors_idx[:mid], neighbors_e_idx[:mid], neighbors_ts[:mid]
            # ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)

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


class Measure:
    '''
    Evaluation of link prediction.
    raw: Given (s, o, t), measurement based on the rank of a true (s, r, o, t) in all possible (s, r, o, t)
    fil: Given (s, o, t), measurement based on the rank of a true (s, r, o, t) in all (s, r, o, t) that don't happen.
    '''
    def __init__(self):
        '''
        mr: mean rank
        mrr: mean reciprocal rank
        '''
        self.hit1 = {"raw": 0.0, "fil": 0.0}
        self.hit3 = {"raw": 0.0, "fil": 0.0}
        self.hit10 = {"raw": 0.0, "fil": 0.0}
        self.mrr = {"raw": 0.0, "fil": 0.0}
        self.mr = {"raw": 0.0, "fil": 0.0}

    def update(self, rank, raw_or_fil):
        if rank == 1:
            self.hit1[raw_or_fil] += 1.0
        if rank <= 3:
            self.hit3[raw_or_fil] += 1.0
        if rank <= 10:
            self.hit10[raw_or_fil] += 1.0

        self.mr[raw_or_fil] += rank
        self.mrr[raw_or_fil] += (1.0 / rank)

    def batch_update(self, rank_l, raw_or_fil):
        '''

        :param rank: [batch_size,]
        :param raw_or_fil:
        :return:
        '''
        self.hit1[raw_or_fil] += np.sum(rank_l == 1)
        self.hit3[raw_or_fil] += np.sum(rank_l <= 3)
        self.hit10[raw_or_fil] += np.sum(rank_l <= 10)
        self.mr[raw_or_fil] += np.sum(rank_l)
        self.mrr[raw_or_fil] += np.reciprocal(rank_l)

    def normalize(self, num_facts):
        for raw_or_fil in ["raw", "fil"]:
            self.hit1[raw_or_fil] /= num_facts
            self.hit3[raw_or_fil] /= num_facts
            self.hit10[raw_or_fil] /= num_facts
            self.mr[raw_or_fil] /= num_facts
            self.mrr[raw_or_fil] /= num_facts

    def print_(self):
        for raw_or_fil in ["raw", "fil"]:
            print(raw_or_fil.title() + " setting:")
            print("\tHit@1 =", self.hit1[raw_or_fil])
            print("\tHit@3 =", self.hit3[raw_or_fil])
            print("\tHit@10 =", self.hit10[raw_or_fil])
            print("\tMR =", self.mr[raw_or_fil])
            print("\tMRR =", self.mrr[raw_or_fil])


def save_config(args, dir:str):
    args_dict = vars(args)
    with open(os.path.join(dir, 'config.json'), 'w') as fp:
        json.dump(args_dict, fp)
