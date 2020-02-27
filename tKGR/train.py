import os
import sys
import argparse
import time
import pdb

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)

from utils import Data, NeighborFinder
from module import TGAN

if torch.cuda.is_available():
    device = 'cuda:4'
else:
    device = 'cpu'

def prepare_inputs(contents, num_neg_sampling=5, dataset='train', start_time=None):
    '''
    :param contents: instance of Data object
    :param num_neg_sampling: how many negtive sampling of objects for each event
    :param start_time: neg sampling for events since start_time (inclusive)
    :param dataset: 'train', 'valid', 'test'
    :return:
    sub_idx_train_t: tensor of subject index [N, ]
    rel_idx_train_t: tensor of relation index [N, ]
    obj_idx_train_t: tensor of object index [N, ]
    neg_obj_idx_train: tensor of negtive sampling of objects [N, num_neg_sampling]
    ts_train_t: tensor of timestamp [N, ]
    '''
    if dataset == 'train':
        contents_dataset = contents.train_data
        if start_time is None:
            start_time = 0
    elif dataset == 'valid':
        contents_dataset = contents.valid_data
        if start_time is None:
            start_time = 5760
        assert start_time >= 5760
    elif dataset == 'test':
        contents_dataset = contents.test_data
        if start_time is None:
            start_time = 6480
        assert start_time >= 6480
    else:
        raise ValueError("invalid input for dataset, choose 'train', 'valid' or 'test'")
    events = np.vstack([np.array(event) for event in contents_dataset if event[3] >= start_time])
    neg_obj_idx = contents.neg_sampling_object(num_neg_sampling, dataset=dataset, start_time=start_time)
    return np.concatenate([events, neg_obj_idx], axis=1)


# help Module for custom Dataloader
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.src_idx = np.array(transposed_data[0])
        self.rel_idx = np.array(transposed_data[1])
        self.obj_idx = np.array(transposed_data[2])
        self.ts = np.array(transposed_data[3])
        self.neg_idx = np.array(transposed_data[4:]).T

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


def val_loss(tgan, valid_dataloader, num_neighbors):
    '''

    :param tgan:
    :param valid_dataloader:
    :param num_neighbors:
    :return:
    '''
    val_loss = []
    with torch.no_grad():
        tgan = tgan.eval()
        for sample in valid_dataloader:
            src_idx = sample.src_idx
            obj_idx = sample.obj_idx
            rel_idx = sample.rel_idx
            ts = sample.ts
            neg_idx = sample.neg_idx

            src_embed, target_embed, neg_embed = tgan.forward(src_idx, obj_idx, neg_idx, ts, num_neighbors)

            rel_idx_t = torch.from_numpy(rel_idx).detach_().to(device)
            rel_embed_diag = torch.diag_embed(model.edge_raw_embed(rel_idx_t))

            loss_pos_term = -torch.nn.LogSigmoid()(
                -torch.bmm(
                    torch.bmm(torch.unsqueeze(src_embed, 1), rel_embed_diag),
                    torch.unsqueeze(target_embed, 2)))  # Bx1
            loss_neg_term = torch.nn.LogSigmoid()(
                torch.bmm(torch.bmm(neg_embed, rel_embed_diag), torch.unsqueeze(src_embed, 2)).view(-1, 1))  # BxQx1

            loss = torch.sum(loss_pos_term) - torch.sum(loss_neg_term)
            val_loss.append(loss.item())
    return np.mean(val_loss)


parser = argparse.ArgumentParser()
parser.add_argument('--num_neg_sampling', type=int, default=5, help="number of negative sampling of objects for each event")
parser.add_argument('--num_layers', type=int, default=2, help='number of TGAN layers')
parser.add_argument('--warm_start_time', type=int, default=1200, help="training data start from what timestamp")
parser.add_argument('--node_feat_dim', type=int, default=100, help='dimension of embedding for node')
parser.add_argument('--edge_feat_dim', type=int, default=100, help='dimension of embedding for edge')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_neighbors', type=int, default=20, help='how many neighbors to aggregate information from, '
                                                                'check paper Inductive Representation Learning '
                                                                'for Temporal Graph for detail')
parser.add_argument('--uniform', action='store_true', help="uniformly sample num_neighbors neighbors")
parser.add_argument('--device', type=int, default=-1, help='-1: cpu, >=0, cuda device')
args = parser.parse_args()

if __name__ == '__main__':
    start_time = time.time()
    structed_time = time.gmtime(start_time)
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.device) if args.device>=0 else 'cpu'
    else:
        device = 'cpu'
    # load dataset
    contents = Data(dataset='ICEWS18_forecasting')

    # init NeighborFinder
    adj_list = contents.get_adj_list()
    nf = NeighborFinder(adj_list)

    # prepare training data
    train_inputs = prepare_inputs(contents, num_neg_sampling=args.num_neg_sampling, start_time=args.warm_start_time)
    # prepare validation data
    val_inputs = prepare_inputs(contents, num_neg_sampling=args.num_neg_sampling, dataset='valid')  # TBD: remove unseen entity and relation in valid and test
    test_inputs = prepare_inputs(contents, num_neg_sampling=args.num_neg_sampling, dataset='test')

    # DataLoader
    train_data_loader = DataLoader(train_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper, pin_memory=False, shuffle=True)
    val_data_loader = DataLoader(val_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper, pin_memory=False, shuffle=True)

    # randomly initialize node and edge feature
    node_feature = np.random.randn(len(adj_list)+1, args.node_feat_dim)  # first row: embedding for dummy node
    # ignore the correlation between relation and reversed relation
    edge_feature = np.random.randn(len(contents.train_data), args.edge_feat_dim)

    model = TGAN(nf, node_feature, edge_feature, num_layers=args.num_layers, device=device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # optimizer

    for epoch in range(args.epoch):
        running_loss = 0.0
        for batch_ndx, sample in tqdm(enumerate(train_data_loader)):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            src_embed, target_embed, neg_embed = model.forward(
                sample.src_idx, sample.obj_idx, sample.neg_idx, sample.ts, num_neighbors=args.num_neighbors)
            sample_rel_idx_t = torch.from_numpy(sample.rel_idx).detach_().to(device)
            rel_embed_diag = torch.diag_embed(model.edge_raw_embed(sample_rel_idx_t))

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
                val_loss = val_loss(model, val_data_loader, num_neighbors=args.num_neighbors)
                print('[%d, %5d] training loss: %.3f, validation loss: %.3f' %
                      (epoch + 1, batch_ndx + 1, running_loss / 2000, val_loss))
                running_loss = 0.0
        CHECKPOINT_PATH = os.path.join(PackageDir, 'Checkpoints', 'checkpoints_{}_{}_{}_{}_{}'.format(
            structed_time.tm_year,
            structed_time.tm_mon,
            structed_time.tm_mday,
            structed_time.tm_hour,
            structed_time.tm_min))

        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'entity_embedding': model.node_raw_embed,
            'relation_embedding': model.edge_raw_embed,
            'time_embedding': model.time_encoder
        }, os.path.join(CHECKPOINT_PATH, 'checkpoint_{}.pt'.format(epoch)))

    print("Finished Training")



