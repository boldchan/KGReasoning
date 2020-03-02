import os
import sys
import time

import numpy as np
import torch
import torch.utils.data import DataLoader

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)

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


def val_loss_acc(tgan, valid_dataloader, num_neighbors, cal_acc:bool=False, sp2o=None, spt2o=None, num_batches=1e8):
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
        for batch_idx, sample in enumerate(valid_dataloader):
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
                neg_label = torch.zeros(neg_embed.shape[0]*neg_embed.shape[1], dtype=torch.float, device=device)

            pos_score = torch.sum(src_embed * rel_embed * target_embed, dim=1)  # [batch_size, ]
            neg_score = torch.sum(torch.unsqueeze(src_embed, 1) * torch.unsqueeze(rel_embed, 1) * neg_embed, dim=2).view(-1)  # [batch_size x num_neg_sampling]

            loss = torch.nn.BCELoss(reduction='sum')(pos_score.sigmoid(), pos_label)
            loss += torch.nn.BCELoss(reduction='sum')(neg_score.sigmoid(), neg_label)
            val_loss += loss.item()

            num_neg_events += len(neg_score)

            # prediction accuracy
            if cal_acc and batch_idx < num_batches:
                for src_idx, rel_idx, obj_idx, ts in list(zip(src_idx_l, rel_idx_l, obj_idx_l, ts_l)):
                    if sp2o is not None:
                        obj_candidate = sp2o[(src_idx, rel_idx)]
                        pred_score = tgan.obj_predict(src_idx, rel_idx, ts, obj_candidate).cpu().numpy()
                        rank = np.sum(pred_score > pred_score[obj_candidate.index(obj_idx)]) + 1
                        measure.update(rank, 'fil')
                    else:
                        pred_score = tgan.obj_predict(src_idx, rel_idx, ts).cpu().numpy()
                        if spt2o is not None:
                            mask = np.ones_like(pred_score, dtype=bool)
                            np.put(mask, spt2o[(src_idx, rel_idx, ts)], False)  # exclude all event with same (s,p,t) even the one with current object
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

if __name__ == '__main__':
    start_time = time.time()
    struct_time = time.gmtime(start_time)
    
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.device) if args.device>=0 else 'cpu'
    else:
        device = 'cpu'
        
    # load dataset
    contents = Data(dataset=args.dataset)
    
    # mapping between (s,p,t) -> o, will be used by evaluating object-prediction
    sp2o = contents.get_sp2o()
    val_spt2o = contents.get_spt2o('valid')
    test_spt2o = contents.get_spt2o('test')
    
    # init NeighborFinder
    adj_list = contents.get_adj_list()
    nf = NeighborFinder(adj_list)
    
    model = TGAN(nf, contents.num_entities, contents.num_relations, args.node_feat_dim, num_layers=args.num_layers, 
                 device=device)
    model.to(device)
    
    # load checkpoint
    checkpoint=torch.load(os.path.join(PackageDir, 'Checkpoints', 'checkpoints_2020_3_1_21_16', 'checkpoint_41.pt'),
                     map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    val_inputs = prepare_inputs(contents, num_neg_sampling=args.num_neg_sampling, dataset='valid')
    val_data_loader = DataLoader(val_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper,
                                         pin_memory=False, shuffle=True)
    if args.evaluation_level == 0:
        val_loss, measure = val_loss_acc(model, val_data_loader,
                                                            num_neighbors=args.num_neighbors,
                                                            cal_acc=True, sp2o=sp2o, spt2o=None)
    elif args.evaluation_level == 1:
        val_loss, measure = val_loss_acc(model, val_data_loader,
                                                            num_neighbors=args.num_neighbors,
                                                            cal_acc=True, sp2o=None, spt2o=val_spt2o)
        
    print('validation loss: %.3f Hit@1: fil %.3f\t raw %.3f, Hit@3: fil %.3f\t raw %.3f, Hit@10: fil %.3f\t raw %.3f, mr: fil %.3f\t raw %.3f, mrr: fil %.3f\t raw %.3f' %
          (val_loss/(num_neg_events + num_events), 
           measure.hit1['fil']/num_events, measure.hit1['raw']/num_events,
           measure.hit3['fil']/num_events, measure.hit3['raw']/num_events,
           measure.hit10['fil']/num_events, measure.hit10['raw']/num_events,
           measure.mr['fil']/num_events, measure.mr['raw']/num_events,
           measure.mrr['fil']/num_events, measure.mrr['raw']/num_events))
    
    
    
    
        