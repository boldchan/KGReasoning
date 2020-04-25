import os
import sys
import time
import argparse

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)
save_dir = '/data/yuwang/tKGR'

from utils import Data, NeighborFinder, Measure, save_config
from module import TGAN

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


def val_loss_acc(tgan, valid_dataloader, num_neighbors, cal_acc: bool = False, sp2o=None, spt2o=None, num_batches=1e8):
    '''

    :param spt2o: if sp2o is None and spt2o is not None, a stricter evaluation on object prediction is performed
    :param sp2o: if sp2o is not None, a looser evaluation on object prediction is performed, in this case spt2o is ignored
    :param tgan:
    :param valid_dataloader:
    :param num_neighbors:
    :param num_batches: how many batches are used to calculate **accuracy**
    :return:
    '''
    val_loss = 0
    measure = Measure()
    eval_level = 0 if sp2o is not None else 1

    with torch.no_grad():
        tgan = tgan.eval()
        num_events = 0
        num_neg_events = 0
        for batch_idx, sample in tqdm(enumerate(valid_dataloader)):
            if batch_idx >= num_batches:
                break
            src_idx_l = sample.src_idx
            obj_idx_l = sample.obj_idx
            rel_idx_l = sample.rel_idx
            ts_l = sample.ts
            neg_idx_l = sample.neg_idx
            num_events += len(src_idx_l)

            src_embed, target_embed, neg_embed = tgan.forward(src_idx_l, obj_idx_l, neg_idx_l, ts_l, num_neighbors)

            rel_idx_t = torch.from_numpy(rel_idx_l).detach_().to(device)
            rel_embed = model.edge_raw_embed(rel_idx_t)

            # rel_embed_diag = torch.diag_embed(rel_embed)
            # loss_pos_term = -torch.nn.LogSigmoid()(
            #     -torch.bmm(
            #         torch.bmm(torch.unsqueeze(src_embed, 1), rel_embed_diag),
            #         torch.unsqueeze(target_embed, 2)))  # Bx1
            # loss_neg_term = torch.nn.LogSigmoid()(
            #     torch.bmm(torch.bmm(neg_embed, rel_embed_diag), torch.unsqueeze(src_embed, 2)).view(-1, 1))  # BxQx1
            #
            # loss = torch.sum(loss_pos_term) - torch.sum(loss_neg_term)
            # val_loss.append(loss.item()/(len()))

            with torch.no_grad():
                pos_label = torch.ones(len(src_embed), dtype=torch.float, device=device)
                neg_label = torch.zeros(neg_embed.shape[0] * neg_embed.shape[1], dtype=torch.float, device=device)

            pos_score = torch.sum(src_embed * rel_embed * target_embed, dim=1)  # [batch_size, ]
            neg_score = torch.sum(torch.unsqueeze(src_embed, 1) * torch.unsqueeze(rel_embed, 1) * neg_embed,
                                  dim=2).view(-1)  # [batch_size x num_neg_sampling]

            loss = torch.nn.BCELoss(reduction='sum')(pos_score.sigmoid(), pos_label)
            loss += torch.nn.BCELoss(reduction='sum')(neg_score.sigmoid(), neg_label)
            val_loss += loss.item()

            num_neg_events += len(neg_score)

            # prediction accuracy
            if cal_acc:
                for src_idx, rel_idx, obj_idx, ts in list(zip(src_idx_l, rel_idx_l, obj_idx_l, ts_l)):
                    if sp2o is not None:
                        obj_candidate = [_ for _ in range(tgan.num_nodes) if _ not in sp2o[(src_idx, rel_idx)]]
                        obj_candidate.append(obj_idx)
                        np.append(obj_candidate, obj_idx)
                        pred_score = tgan.obj_predict(src_idx, rel_idx, ts, obj_candidate).cpu().numpy()
                        rank = np.sum(pred_score > pred_score[obj_candidate.index(obj_idx)]) + 1
                        measure.update(rank, 'fil')
                    else:
                        pred_score = tgan.obj_predict(src_idx, rel_idx, ts).cpu().numpy()
                        if spt2o is not None:
                            mask = np.ones_like(pred_score, dtype=bool)
                            np.put(mask, spt2o[(src_idx, rel_idx, ts)],
                                   False)  # exclude all event with same (s,p,t) even the one with current object
                            rank = np.sum(pred_score[mask] > pred_score[obj_idx]) + 1
                            measure.update(rank, 'fil')
                        rank = np.sum(pred_score > pred_score[obj_idx]) + 1  # int
                        measure.update(rank, 'raw')
            print('[evaluation level: %d]validation loss: %.3f Hit@1: fil %.3f\t raw %.3f, Hit@3: fil %.3f\t raw %.3f, Hit@10: fil %.3f\t raw %.3f, mr: fil %.3f\t raw %.3f, mrr: fil %.3f\t raw %.3f' %
                  (eval_level, val_loss/(num_neg_events + num_events),
                   measure.hit1['fil']/num_events, measure.hit1['raw']/num_events,
                   measure.hit3['fil']/num_events, measure.hit3['raw']/num_events,
                   measure.hit10['fil']/num_events, measure.hit10['raw']/num_events,
                   measure.mr['fil']/num_events, measure.mr['raw']/num_events,
                   measure.mrr['fil']/num_events, measure.mrr['raw']/num_events))

        measure.normalize(num_events)
        val_loss /= (num_neg_events + num_events)
    return val_loss, measure


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ICEWS18_forecasting', help='specify data set')
parser.add_argument('--num_neg_sampling', type=int, default=5,
                    help="number of negative sampling of objects for each event")
parser.add_argument('--num_layers', type=int, default=2, help='number of TGAN layers')
parser.add_argument('--node_feat_dim', type=int, default=100, help='dimension of embedding for node')
parser.add_argument('--edge_feat_dim', type=int, default=100, help='dimension of embedding for edge')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_neighbors', type=int, default=20, help='how many neighbors to aggregate information from, '
                                                                  'check paper Inductive Representation Learning '
                                                                  'for Temporal Graph for detail')
parser.add_argument('--sampling', type=int, default=1, help="neighborhood sampling strategy, 0: uniform, 1: first num_neighbors, 2: last num_neighbors")
parser.add_argument('--device', type=int, default=-1, help='-1: cpu, >=0, cuda device')
parser.add_argument('--val_num_batch', type=int, default=1e8,
                    help='how many validation batches are used for calculating accuracy '
                         'specify a really large integer to use all validation set')
parser.add_argument('--evaluation_level', type=int, default=1, choices=[0, 1],
                    help="0: a looser 'fil' evaluation on object prediction,"
                         "prediction score is ranked among objects that "
                         "don't exist in whole data set. sp2o will be used"
                         "1: a stricter 'fil' evaluation on object prediction"
                         "prediction score is ranked among objects that"
                         "don't exist in the current timestamp. spt2o is used")
parser.add_argument('--checkpoint_dir', type=str, help='directory of checkpoint')
parser.add_argument('--checkpoint_ind', type=int, help='index indicates the epoch of checkpoint')
parser.add_argument('--eval_one_epoch', action='store_true', help="set to only evaluate the epoch specified by "
                                                                  "checkpoint_ind, used for debugging")
parser.add_argument('--eval_num_batches', type=int, default=128, help='number of batches to perform evaluation')
parser.add_argument('--add_reverse', action='store_true', default=None)

args = parser.parse_args()
if __name__ == '__main__':
    start_time = time.time()
    struct_time = time.gmtime(start_time)

    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.device) if args.device >= 0 else 'cpu'
    else:
        device = 'cpu'

    # load data set
    contents = Data(dataset=args.dataset, add_reverse_relation=args.add_reverse)

    # mapping between (s,p,t) -> o, will be used by evaluating object-prediction
    sp2o = contents.get_sp2o()
    val_spt2o = contents.get_spt2o('valid')
    test_spt2o = contents.get_spt2o('test')

    # init NeighborFinder
    adj = contents.get_adj_list() if args.add_reverse else contents.get_adj_dict()
    max_time = max(contents.data[:, 3])
    nf = NeighborFinder(adj, sampling=args.sampling, max_time=max_time, num_entities=len(contents.id2entity))

    model = TGAN(nf, contents.num_entities, contents.num_relations, args.node_feat_dim, num_layers=args.num_layers,
                 device=device)
    model.to(device)
    val_inputs = prepare_inputs(contents, num_neg_sampling=args.num_neg_sampling, dataset='valid')

    if args.eval_one_epoch:
        checkpoint_to_eval = [args.checkpoint_ind]
    else:
        checkpoint_to_eval = [_ for _ in range(args.checkpoint_ind)]
    for check_idx in checkpoint_to_eval:
        # load checkpoint
        checkpoint = torch.load(os.path.join(save_dir, 'Checkpoints', args.checkpoint_dir,
                                             'checkpoint_{}.pt'.format(check_idx)), map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        val_data_loader = DataLoader(val_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper,
                                     pin_memory=False, shuffle=True)

        model.eval()
        if args.evaluation_level == 0:
            val_loss, measure = val_loss_acc(model, val_data_loader,
                                             num_neighbors=args.num_neighbors,
                                             cal_acc=True, sp2o=sp2o, spt2o=None,
                                             num_batches=args.eval_num_batches)
        elif args.evaluation_level == 1:
            val_loss, measure = val_loss_acc(model, val_data_loader,
                                             num_neighbors=args.num_neighbors,
                                             cal_acc=True, sp2o=None, spt2o=val_spt2o,
                                             num_batches=args.eval_num_batches)
        else:
            raise ValueError("evaluation level must be 0 or 1")

        print('[checkpoint_%d] validation loss: %.3f Hit@1: fil %.3f\t raw %.3f, Hit@3: fil %.3f\t raw %.3f, '
              'Hit@10: fil %.3f\t raw %.3f, mr: fil %.3f\t raw %.3f, mrr: fil %.3f\t raw %.3f' %
              (check_idx, val_loss,
               measure.hit1['fil'], measure.hit1['raw'],
               measure.hit3['fil'], measure.hit3['raw'],
               measure.hit10['fil'], measure.hit10['raw'],
               measure.mr['fil'], measure.mr['raw'],
               measure.mrr['fil'], measure.mrr['raw']))
