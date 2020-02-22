import os
import sys
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

PackageDir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tKGR')
sys.path.insert(1, PackageDir)

from utils import Data, NeighborFinder
from module import TGAN

if torch.cuda.is_available():
    device = 'cuda:6'
else:
    device = 'cpu'

def prepare_inputs(contents, num_neg_sampling=5, start_time=0):
    '''
    :param contents: instance of Data object
    :param num_neg_sampling: how many negtive sampling of objects for each event
    :param start_time: neg sampling for events since start_time (inclusive)
    :return:
    sub_idx_train_t: tensor of subject index [N, ]
    rel_idx_train_t: tensor of relation index [N, ]
    obj_idx_train_t: tensor of object index [N, ]
    neg_obj_idx_train: tensor of negtive sampling of objects [N, num_neg_sampling]
    ts_train_t: tensor of timestamp [N, ]
    '''
    sub_idx_train = np.array([event[0] for event in contents.train_data if event[3] >= start_time])
    rel_idx_train = np.array([event[1] for event in contents.train_data if event[3] >= start_time])
    obj_idx_train = np.array([event[2] for event in contents.train_data if event[3] >= start_time])
    ts_train = np.array([event[3] for event in contents.train_data if event[3] >= start_time])
    neg_obj_idx_train = contents.neg_sampling_object(num_neg_sampling, start_time=start_time)

    sub_idx_train_t = torch.from_numpy(sub_idx_train).long()
    rel_idx_train_t = torch.from_numpy(rel_idx_train).long()
    obj_idx_train_t = torch.from_numpy(obj_idx_train).long()
    neg_obj_idx_train_t = torch.from_numpy(neg_obj_idx_train).long()
    ts_train_t = torch.from_numpy(ts_train).long()
    return sub_idx_train_t, rel_idx_train_t, obj_idx_train_t, neg_obj_idx_train_t, ts_train_t


# help Module for custom Dataloader
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.src_idx = torch.stack(transposed_data[0], 0)
        self.rel_idx = torch.stack(transposed_data[1], 0)
        self.obj_idx = torch.stack(transposed_data[2], 0)
        self.neg_idx = torch.stack(transposed_data[3], 0)
        self.ts = torch.stack(transposed_data[4], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.src_idx = self.src_idx.pin_memory()
        self.rel_idx = self.rel_idx.pin_memory()
        self.obj_idx = self.obj_idx.pin_memory()
        self.neg_idx = self.neg_idx.pin_memory()
        self.ts = self.ts.pin_memory()

        return self


# help function for custom Dataloader
def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

parser = argparse.ArgumentParser()
parser.add_argument('num_neg_sammpling', type=int, help="number of negative sampling of objects for each event")
parser.add_argument('warm_start_time', type=int, help="training data start from what timestamp")
parser.add_argument('node_feat_dim', type=int, default=100, help='dimension of embedding for node')
parser.add_argument('edge_feat_dim', type=int, default=100, help='dimension of embedding for edge')
parser.add_argument('lr', type=float, default=0.001)
parser.add_argument('epoch', type=int, default=5)
parser.add_argument('batch_size', type=int, default=10)
parser.add_argument('num_neighbors', type=int, default=20, help='how many neighbors to aggregate information from, '
                                                                'check paper Inductive Representation Learning '
                                                                'for Temporal Graph for detail')
args = parser.parse_args()

if __file__ == '__main__':
    # load dataset
    contents = Data(dataset='ICEWS18_forecasting')

    # init NeighborFinder
    adj_list = contents.get_adj_list()
    nf = NeighborFinder(adj_list)

    # prepare training data
    sub_idx_train_t, rel_idx_train_t, obj_idx_train_t, neg_obj_idx_train_t, ts_train_t = prepare_inputs(
        contents, num_neg_sampling=args.num_neg_sampling, start_time=args.warm_start_time)

    # DataLoader
    src_and_t = TensorDataset(sub_idx_train_t, rel_idx_train_t, obj_idx_train_t, neg_obj_idx_train_t, ts_train_t)
    train_data_loader = DataLoader(src_and_t, batch_size=args.batch_size, collate_fn=collate_wrapper, pin_memory=True, shuffle=True)

    # check if data is on GPU
    for sample in train_data_loader:
        print("Data is pinned in GPU? :{}".format(sample.ts.is_pinned()))
        break

    # randomly initialize node and edge feature
    node_feature = np.random.randn(len(adj_list), args.node_feat_dim)
    # ignore the correlation between relation and reversed relation
    edge_feature = np.random.randn(len(contents.train_data), args.edge_feat_dim)

    model = TGAN(nf, node_feature, edge_feature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # optimizer

    for epoch in range(args.epoch):
        running_loss = 0.0
        for batch_ndx, sample in enumerate(train_data_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            src_embed, target_embed, neg_embed = model.forward(
                sample.src_idx, sample.obj_idx, sample.neg_idx, sample.ts, num_neighbors=args.num_neighbors)
            rel_embed_diag = torch.diag_embed(model.edge_raw_embed(sample.rel_idx))

            loss_pos_term = -torch.nn.LogSigmoid()(
                -torch.bmm(
                    torch.bmm(torch.unsqueeze(src_embed, 1), rel_embed_diag),
                    torch.unsqueeze(target_embed, 2)))  # Bx1
            loss_neg_term = torch.nn.LogSigmoid()(
                torch.bmm(torch.bmm(neg_embed, rel_embed_diag), torch.unsqueeze(src_embed, 2)).view(-1, 1))  # BxQx1
            loss = torch.sum(loss_pos_term) - torch.sum(loss_neg_term)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if batch_ndx % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_ndx + 1, running_loss / 2000))
                running_loss = 0.0

    print("Finished Training")


