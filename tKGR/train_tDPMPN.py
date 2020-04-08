import os
import sys

from collections import defaultdict
import argparse
import time
import copy
import pdb

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)

from utils import Data, NeighborFinder, Measure, save_config
from module import tDPMPN
import config
import local_config

save_dir = local_config.save_dir

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def prepare_inputs(contents, num_neg_sampling=5, dataset='train', start_time=0):
    '''
    :param contents: instance of Data object
    :param num_neg_sampling: how many negtive sampling of objects for each event
    :param start_time: neg sampling for events since start_time (inclusive)
    :param dataset: 'train', 'valid', 'test'
    :return:
    events concatenated with negative sampling
    '''
    if dataset == 'train':
        contents_dataset = contents.train_data
        assert start_time < max(contents_dataset[:, 3])
    elif dataset == 'valid':
        contents_dataset = contents.valid_data_seen_entity
        assert start_time < max(contents_dataset[:, 3])
    elif dataset == 'test':
        contents_dataset = contents.test_data_seen_entity
        assert start_time < max(contents_dataset[:, 3])
    else:
        raise ValueError("invalid input for dataset, choose 'train', 'valid' or 'test'")
    events = np.vstack([np.array(event) for event in contents_dataset if event[3] >= start_time])
    neg_obj_idx = contents.neg_sampling_object(num_neg_sampling, dataset=dataset, start_time=start_time)
    return np.concatenate([events, neg_obj_idx], axis=1)


# help Module for custom Dataloader
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.src_idx = np.array(transposed_data[0], dtype=np.int32)
        self.rel_idx = np.array(transposed_data[1], dtype=np.int32)
        self.target_idx = np.array(transposed_data[2], dtype=np.int32)
        self.ts = np.array(transposed_data[3], dtype=np.int32)
        self.neg_idx = np.array(transposed_data[4:-1], dtype=np.int32).T
        self.event_idx = np.array(transposed_data[-1], dtype=np.int32)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.src_idx = self.src_idx.pin_memory()
        self.rel_idx = self.rel_idx.pin_memory()
        self.target_idx = self.target_idx.pin_memory()
        self.neg_idx = self.neg_idx.pin_memory()
        self.ts = self.ts.pin_memory()
        self.event_idx = self.event_idx.pin_memory()

        return self


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


def segment_topk(t, segment_idx, k, sorted=False):
    """
    compute topk along segments of a tensor
    params:
        t: Tensor, 1d, dtype=torch.float32
        segment_idx: numpy.array, 1d, dtype=numpy.int32, sorted
        k: k largest values
    return:
        values[i]: Tensor of topk of segment i
        indices[i]: numpy.array of position of topk elements of segment i in original Tensor t
    """
    mask = segment_idx[1:] != segment_idx[:-1]
    key_idx = np.concatenate([np.array([0], dtype=np.int32),
                             np.arange(1, len(segment_idx))[mask],
                             np.array([len(segment_idx)])])
    values = []
    indices = []
    for s, e in zip(key_idx[:-1], key_idx[1:]):
        if e - s < k:
            if sorted:
                sorted_value, sorted_indices = torch.sort(t[s:e], descending=True)
                values.append(sorted_value)
                indices.append(s + sorted_indices.cpu().numpy())
            else:
                values.append(t[s:e])
                indices.append(np.arange(s, e))
        else:
            segment_values, segment_indices = torch.topk(t[s:e], k, sorted=sorted)
            values.append(segment_values)
            indices.append(s + segment_indices.cpu().numpy())
    return values, indices


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None, help='specify data set')
parser.add_argument('--num_neg_sampling', type=int, default=5,
                    help="number of negative sampling of objects for each event")
parser.add_argument('--tgan_num_layers', type=int, default=2, help='number of TGAN layers')
parser.add_argument('--warm_start_time', type=int, default=48, help="training data start from what timestamp")
parser.add_argument('--emb_dim', type=int, default=128, help='dimension of embedding for node, realtion and time')
parser.add_argument('--emb_dim_sm', type=int, default=32, help='smaller dimension of embedding, '
                                                               'ease the computation of attention for attending from horizon')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_neighbors', type=int, default=40, help='how many neighbors to aggregate information from, '
                                                                  'check paper Inductive Representation Learning '
                                                                  'for Temporal Graph for detail')
parser.add_argument('--device', type=int, default=-1, help='-1: cpu, >=0, cuda device')
parser.add_argument('--sampling', type=int, default=2, help='strategy to sample neighbors, 0: uniform, 1: first num_neighbors, 2: last num_neighbors')
parser.add_argument('--DP_steps', type=int, default=2, help='number of DP steps')
parser.add_argument('--max_attended_nodes', type=int, default=20, help='max number of nodes in attending from horizon')
args = parser.parse_args()

if __name__ == "__main__":
    # check cuda
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.device) if args.device >= 0 else 'cpu'
    else:
        device = 'cpu'

    # construct NeighborFinder
    contents = Data(dataset=args.dataset, add_reverse_relation=False)
    adj = contents.get_adj_dict()
    max_time = max(contents.data[:, 3])
    nf = NeighborFinder(adj, sampling=args.sampling, max_time=max_time, num_entities=len(contents.id2entity))

    # load data
    train_inputs = prepare_inputs(contents, num_neg_sampling=args.num_neg_sampling, start_time=args.warm_start_time)
    train_data_loader = DataLoader(train_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper, pin_memory=False, shuffle=True)

    val_inputs = prepare_inputs(contents, num_neg_sampling=args.num_neg_sampling, dataset='valid')
    val_data_loader = DataLoader(val_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper, pin_memory=False, shuffle=True)

    # construct model
    model = tDPMPN(nf, len(contents.id2entity), len(contents.id2relation), args.emb_dim, args.emb_dim_sm)
    # move a model to GPU before constructing an optimizer, http://pytorch.org/docs/master/optim.html
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch = 0
    train_loss = 0.
    accuracy = 0.

    for batch_ndx, sample in enumerate(train_data_loader):
        optimizer.zero_grad()
        model.zero_grad()
        model.train()
        
        src_idx_l, rel_idx_l, target_idx_l, cut_time_l = sample.src_idx, sample.rel_idx, sample.target_idx, sample.ts
        model.set_init(src_idx_l, rel_idx_l, target_idx_l, cut_time_l, batch_ndx+1, epoch)
        query_src_emb, query_rel_emb, query_time_emb, attending_nodes, attending_node_attention, memorized_embedding = \
            model.initialize()
        
        # query_time_emb.to(device)
        # query_src_emb.to(device)
        # query_rel_emb.to(device)
        # attending_node_attention.to(device)

        for step in range(args.DP_steps):
            print("{}-th DP step".format(step))
            attending_nodes, attending_node_attention, memorized_embedding = \
                model.flow(attending_nodes, attending_node_attention, memorized_embedding, query_src_emb, query_rel_emb, query_time_emb)
        entity_att_score, entities = model.get_entity_attn_score(attending_node_attention, attending_nodes)
        one_hot_label = torch.from_numpy(np.array([int(v==target_idx_l[eg_idx]) for eg_idx, v in entities], dtype=np.float32)).to(device)
        loss = torch.nn.BCELoss()(entity_att_score, one_hot_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        epoch += 1

        if epoch % 5 == 4:
            hit_1 = hit_3 = hit_10 = 0
            num_query = 0

            for batch_ndx, sample in enumerate(val_data_loader):
                model.eval()
                
                src_idx_l, rel_idx_l, target_idx_l, cut_time_l = sample.src_idx, sample.rel_idx, sample.target_idx, sample.ts
                num_query += len(src_idx_l)
                
                model.set_init(src_idx_l, rel_idx_l, target_idx_l, cut_time_l, batch_ndx+1, 0)
                query_src_emb, query_rel_emb, query_time_emb, attending_nodes, attending_node_attention, memorized_embedding = model.initialize()
                for step in range(args.DP_steps):
                    attending_nodes, attending_node_attention, memorized_embedding = \
                        model.flow(attending_nodes, attending_node_attention, memorized_embedding, query_src_emb, query_rel_emb, query_time_emb)
                entity_att_score, entities = model.get_entity_attn_score(attending_node_attention, attending_nodes)
                
                _, indices = segment_topk(entity_att_score, entities[:, 0], 10, sorted=True)
                for i, target in enumerate(target_idx_l):
                    top10 = entities[indices[i]]
                    hit_1 += target == top10[0,1]
                    hit_3 += target in top10[:3, 1]
                    hit_10 += target in top10[:, 1]
            print("hit@1: {}, hit@3: {}, hit@10: {}".format(hit_1/num_query, hit_3/num_query, hit_10/num_query))
