import os
# the mode argument of the os.makedirs function may be ignored on some systems
# umask (user file-creation mode mask) specify the default denial value of variable mode, 
# which means if this value is passed to makedirs function,  
# it will be ignored and a folder/file with d_________ will be created 
# we can either set the umask or specify mode in makedirs

# oldmask = os.umask(0o770)

import sys
import gc

from collections import defaultdict
import argparse
import time
import copy
import pdb
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)

from utils import Data, NeighborFinder, Measure, save_config, get_git_version_short_hash
from module import tDPMPN, segment_softmax_op_v2
import config
import local_config
from segment import *
from database_op import create_connection, create_table

# from gpu_profile import gpu_profile

save_dir = local_config.save_dir

# Reproducibility
torch.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_checkpoint(model, optimizer, checkpoint_dir):
    if os.path.isfile(checkpoint_dir):
        print("=> loading checkpoint '{}'".format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_dir, checkpoint['epoch']))
    else:
        raise IOError("=> no checkpoint found at '{}'".format(checkpoint_dir))

    return model, optimizer, start_epoch


def reset_time_cost():
    return {'model': defaultdict(float), 'graph': defaultdict(float), 'grad': defaultdict(float),
            'data': defaultdict(float)}


def str_time_cost(tc):
    if tc is not None:
        data_tc = ', '.join('data.{} {:3f}'.format(k, v) for k, v in tc['data'].items())
        model_tc = ', '.join('m.{} {:3f}'.format(k, v) for k, v in tc['model'].items())
        graph_tc = ', '.join('g.{} {:3f}'.format(k, v) for k, v in tc['graph'].items())
        grad_tc = ', '.join('d.{} {:3f}'.format(k, v) for k, v in tc['grad'].items())
        return model_tc + ', ' + graph_tc + ', ' + grad_tc
    else:
        return ''


def prepare_inputs(contents, num_neg_sampling=5, dataset='train', start_time=0, tc=None):
    '''
    :param tc: time recorder
    :param contents: instance of Data object
    :param num_neg_sampling: how many negtive sampling of objects for each event
    :param start_time: neg sampling for events since start_time (inclusive)
    :param dataset: 'train', 'valid', 'test'
    :return:
    events concatenated with negative sampling
    '''
    t_start = time.time()
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
    if args.timer:
        tc['data']['load_data'] += time.time() - t_start
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


def segment_rank(t, entities, target_idx_l):
    """
    compute rank of ground truth (target_idx_l) in prediction according to score, i.e. t
    :param t: prediction score
    :param entities: 2-d numpy array, (segment_idx, entity_idx)
    :param target_idx_l: 1-d numpy array, (batch_size, )
    :return:
    """
    mask = entities[1:, 0] != entities[:-1, 0]
    key_idx = np.concatenate([np.array([0], dtype=np.int32),
                              np.arange(1, len(entities))[mask],
                              np.array([len(entities)])])
    rank = []
    found_mask = []
    for i, (s, e) in enumerate(zip(key_idx[:-1], key_idx[1:])):
        arg_target = np.nonzero(entities[s:e, 1] == target_idx_l[i])[0]
        if arg_target.size > 0:
            found_mask.append(True)
            rank.append(torch.sum(t[s:e] > t[s:e][torch.from_numpy(arg_target)]).item() + 1)
        else:
            found_mask.append(False)
            rank.append(1e9)  # MINERVA set rank to +inf if not in path, we follow this scheme
    return np.array(rank), found_mask


def segment_rank_fil(t, entities, target_idx_l, sp2o, queries_sub, queries_pre, spt2o):
    """
    compute rank of ground truth (target_idx_l) in prediction according to score, i.e. t
    :param queries_pre: 1d numpy array of query predicate
    :param queries_sub: 1d numpy array of query subject
    :param sp2o:
    :param t: prediction score
    :param entities: 2-d numpy array, (segment_idx, entity_idx)
    :param target_idx_l: 1-d numpy array, (batch_size, )
    :return:
    """
    mask = entities[1:, 0] != entities[:-1, 0]
    key_idx = np.concatenate([np.array([0], dtype=np.int32),
                              np.arange(1, len(entities))[mask],
                              np.array([len(entities)])])
    rank = []
    rank_fil = []
    rank_fil_t = []
    found_mask = []
    for i, (s, e) in enumerate(zip(key_idx[:-1], key_idx[1:])):
        arg_target = np.nonzero(entities[s:e, 1] == target_idx_l[i])[0]
        if arg_target.size > 0:
            found_mask.append(True)
            sub, pre = queries_sub[i], queries_pre[i]
            obj_exist = sp2o[(sub, pre)]
            obj_exist_t = spt2o[(sub, pre)]
            rank_pred_com1 = torch.sum(t[s:e] > t[s:e][torch.from_numpy(arg_target)]).item()
            rank_pred_com2 = torch.sum(t[s:e] == t[s:e][torch.from_numpy(arg_target)]).item()
            rank.append(rank_pred_com1 + ((rank_pred_com2 - 1) / 2) + 1)
            fil = [ent not in obj_exist for ent in entities[s:e, 1]]
            fil_t = [ent not in obj_exist_t for ent in entities[s:e, 1]]
            rank_pred_com1_fil = torch.sum(t[s:e][fil] > t[s:e][torch.from_numpy(arg_target)]).item()
            rank_pred_com2_fil = torch.sum(t[s:e][fil] == t[s:e][torch.from_numpy(arg_target)]).item()
            rank_fil.append(rank_pred_com1_fil + ((rank_pred_com2_fil - 1) / 2) + 1)
            rank_pred_com1_fil_t = torch.sum(t[s:e][fil_t] > t[s:e][torch.from_numpy(arg_target)]).item()
            rank_pred_com2_fil_t = torch.sum(t[s:e][fil_t] == t[s:e][torch.from_numpy(arg_target)]).item()
            rank_fil_t.append(rank_pred_com1_fil_t + ((rank_pred_com2_fil_t - 1) / 2) + 1)
        else:
            found_mask.append(False)
            rank.append(1e9)  # MINERVA set rank to +inf if not in path, we follow this scheme
            rank_fil.append(1e9)
    return np.array(rank), found_mask, np.array(rank_fil), np.array([rank_fil_t])


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
parser.add_argument('--tgan_num_neighbors', type=int, default=40,
                    help='how many neighbors to aggregate information from, '
                         'check paper Inductive Representation Learning '
                         'for Temporal Graph for detail')
parser.add_argument('--device', type=int, default=-1, help='-1: cpu, >=0, cuda device')
parser.add_argument('--sampling', type=int, default=2,
                    help='strategy to sample neighbors, 0: uniform, 1: first num_neighbors, 2: last num_neighbors')
parser.add_argument('--DP_steps', type=int, default=2, help='number of DP steps')
parser.add_argument('--DP_num_neighbors', type=int, default=20, help='number of neighbors sampled for sampling horizon')
parser.add_argument('--max_attended_nodes', type=int, default=20, help='max number of nodes in attending from horizon')
parser.add_argument('--add_reverse', action='store_true', default=False, help='add reverse relation into data set')
parser.add_argument('--load_checkpoint', type=str, default=None, help='train from checkpoints')
parser.add_argument('--timer', action='store_true', default=None, help='set to profile time consumption for some func')
parser.add_argument('--use_TGAN', action='store_true', default=None, help='use hidden representation of TGAN')
parser.add_argument('--debug', action='store_true', default=None, help='in debug mode, checkpoint will not be saved')
parser.add_argument('--sqlite', action='store_true', default=None, help='save information to sqlite')
parser.add_argument('--weight_factor', type=float, default=1, help='sampling 3, scale weight')
args = parser.parse_args()

if __name__ == "__main__":
    print(args)
    # sys.settrace(gpu_profile)
    # check cuda
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.device) if args.device >= 0 else 'cpu'
    else:
        device = 'cpu'

    if args.sqlite and not args.debug:
        sqlite_conn = create_connection(os.path.join(save_dir, 'tKGR.db'))
        task_col = ('checkpoint_dir', 'dataset', 'emb_dim', 'emb_dim_sm', 'lr', 'batch_size', 'sampling', 'DP_steps',
                    'DP_num_neighbors', 'max_attended_nodes', 'add_reverse', 'git_hash')
        sql_create_tasks_table = """
        CREATE TABLE IF NOT EXISTS tasks (
        checkpoint_dir text PRIMARY KEY,
        dataset text NOT NULL,
        emb_dim integer NOT NULL,
        emb_dim_sm integer NOT NULL,
        lr real NOT NULL,
        batch_size integer NOT NULL,
        sampling integer NOT NULL,
        DP_steps integer NOT NULL,
        DP_num_neighbors integer NOT NULL,
        max_attended_nodes integer NOT NULL,
        add_reverse integer NOT NULL,
        git_hash text NOT NULL);
        """
        logging_col = (
            'checkpoint_dir', 'epoch', 'training_loss', 'validation_loss', 'HITS_1_raw', 'HITS_3_raw', 'HITS_10_raw',
            'HITS_INF', 'MRR_raw', 'HITS_1_fil', 'HITS_3_fil', 'HITS_10_fil', 'MRR_fil')
        sql_create_loggings_table = """ CREATE TABLE IF NOT EXISTS logging (
        checkpoint_dir text NOT NULL, 
        epoch integer NOT NULL,
        training_loss real,
        validation_loss real,
        HITS_1_raw real,
        HITS_3_raw real,
        HITS_10_raw real,
        HITS_INF real,
        MRR_raw real,
        HITS_1_fil real,
        HITS_3_fil real,
        HITS_10_fil real,
        MRR_fil real,
        PRIMARY KEY (checkpoint_dir, epoch),
        FOREIGN KEY (checkpoint_dir) REFERENCES tasks (checkpoint_dir)
        );"""
        if sqlite_conn is not None:
            create_table(sqlite_conn, sql_create_tasks_table)
            create_table(sqlite_conn, sql_create_loggings_table)
        else:
            print("Error! cannot create the database connection.")

    time_cost = None
    if args.timer:
        time_cost = reset_time_cost()

    # save configuration
    start_time = time.time()
    struct_time = time.gmtime(start_time)

    if not args.debug:
        if args.load_checkpoint is None:
            checkpoint_dir = 'checkpoints_{}_{}_{}_{}_{}_{}'.format(
                struct_time.tm_year,
                struct_time.tm_mon,
                struct_time.tm_mday,
                struct_time.tm_hour,
                struct_time.tm_min,
                struct_time.tm_sec)
            CHECKPOINT_PATH = os.path.join(save_dir, 'Checkpoints', checkpoint_dir)
            if not os.path.exists(CHECKPOINT_PATH):
                os.makedirs(CHECKPOINT_PATH, mode=0o770)
        else:
            checkpoint_dir = os.path.dirname(args.load_checkpoint)
            CHECKPOINT_PATH = os.path.join(save_dir, 'Checkpoints', os.path.dirname(args.load_checkpoint))

        if args.sqlite:
            args_dict = vars(args)
            git_hash = get_git_version_short_hash()
            args_dict['checkpoint_dir'] = checkpoint_dir
            args_dict['git_hash'] = git_hash
            args_dict['add_reverse'] = int(args_dict['add_reverse'])
            with sqlite_conn:
                placeholders = ', '.join('?' * len(task_col))
                sql_hp = 'INSERT OR IGNORE INTO tasks({}) VALUES ({})'.format(', '.join(task_col), placeholders)
                cur = sqlite_conn.cursor()
                cur.execute(sql_hp, [args_dict[col] for col in task_col])
                task_id = cur.lastrowid
            print("Log configuration to SQLite under {}".format(os.path.join(save_dir, 'tKGR.db')))
        else:
            print("Save checkpoints under {}".format(CHECKPOINT_PATH))
            save_config(args, CHECKPOINT_PATH)
            print("Log configuration under {}".format(CHECKPOINT_PATH))

    # construct NeighborFinder
    if args.timer:
        t_start = time.time()
    contents = Data(dataset=args.dataset, add_reverse_relation=args.add_reverse)

    sp2o = contents.get_sp2o()
    val_spt2o = contents.get_spt2o('valid')

    adj = contents.get_adj_dict()
    max_time = max(contents.data[:, 3])
    nf = NeighborFinder(adj, sampling=args.sampling, max_time=max_time, num_entities=len(contents.id2entity), weight_factor=args.weight_factor)
    if args.timer:
        time_cost['data']['ngh'] = time.time() - t_start

    # construct model
    model = tDPMPN(nf, len(contents.id2entity), len(contents.id2relation), args.emb_dim, args.emb_dim_sm,
                   DP_num_neighbors=args.DP_num_neighbors, tgan_num_neighbors=args.tgan_num_neighbors, device=device,
                   use_TGAN=args.use_TGAN)
    # move a model to GPU before constructing an optimizer, http://pytorch.org/docs/master/optim.html
    model.to(device)
    model.TGAN.node_raw_embed.cpu()
    model.TGAN.edge_raw_embed.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0

    if args.load_checkpoint is not None:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer,
                                                        os.path.join(save_dir, 'Checkpoints', args.load_checkpoint))
        start_epoch += 1

    for epoch in range(start_epoch, args.epoch):
        print("epoch: ", epoch)
        # load data
        train_inputs = prepare_inputs(contents, num_neg_sampling=args.num_neg_sampling,
                                      start_time=args.warm_start_time, tc=time_cost)
        train_data_loader = DataLoader(train_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper,
                                       pin_memory=False, shuffle=True)

        running_loss = 0.

        for batch_ndx, sample in tqdm(enumerate(train_data_loader)):
            optimizer.zero_grad()
            model.zero_grad()
            model.train()

            src_idx_l, rel_idx_l, target_idx_l, cut_time_l = sample.src_idx, sample.rel_idx, sample.target_idx, sample.ts
            model.set_init(src_idx_l, rel_idx_l, target_idx_l, cut_time_l, batch_ndx + 1, epoch)
            query_src_emb, query_rel_emb, query_time_emb, attending_nodes, attending_node_attention, memorized_embedding = \
                model.initialize()

            # query_time_emb.to(device)
            # query_src_emb.to(device)
            # query_rel_emb.to(device)
            # attending_node_attention.to(device)

            for step in range(args.DP_steps):
                # print("{}-th DP step".format(step))
                attending_nodes, attending_node_attention, memorized_embedding, _ = \
                    model.flow(attending_nodes, attending_node_attention, memorized_embedding, query_src_emb,
                               query_rel_emb, query_time_emb, tc=time_cost)
            entity_att_score, entities = model.get_entity_attn_score(attending_node_attention, attending_nodes,
                                                                     tc=time_cost)

            # l1 norm on entity attention
            entity_att_score = segment_norm_l1(entity_att_score, entities[:, 0])

            one_hot_label = torch.from_numpy(
                np.array([int(v == target_idx_l[eg_idx]) for eg_idx, v in entities], dtype=np.float32)).to(device)
            try:
                loss = torch.nn.BCELoss()(entity_att_score, one_hot_label)
            except:
                print(entity_att_score)
                entity_att_score_np = entity_att_score.detach().numpy()
                print("all entity score smaller than 1:", all(entity_att_score_np < 1))
                print("all entity score greater than 0:", all(entity_att_score_np > 0))
                raise ValueError("Check if entity score in (0,1)")
            if args.timer:
                t_start = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if args.timer:
                time_cost['grad']['comp'] += time.time() - t_start

            if args.timer:
                t_start = time.time()
            optimizer.step()
            if args.timer:
                time_cost['grad']['apply'] += time.time() - t_start

            running_loss += loss.item()

            # if batch_ndx % 1 == 0:
            #     print('[%d, %5d] training loss: %.3f' % (epoch, batch_ndx, running_loss / 1))
            #     running_loss = 0.0
            print(str_time_cost(time_cost))
            if args.timer:
                time_cost = reset_time_cost()

            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             print(type(obj), obj.size())
            #     except:
            #         pass
        running_loss /= batch_ndx + 1

        model.eval()
        if not args.debug:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(CHECKPOINT_PATH, 'checkpoint_{}.pt'.format(epoch)))

        if epoch % 1 == 0:
            hit_1 = hit_3 = hit_10 = 0
            hit_1_fil = hit_3_fil = hit_10_fil = 0
            hit_1_fil_t = hit_3_fil_t = hit_10_fil_t = 0
            found_cnt = 0
            MR_total = 0
            MR_found = 0
            MRR_total = 0
            MRR_found = 0
            MRR_total_fil = 0
            MRR_total_fil_t = 0
            num_query = 0
            mean_degree = 0
            mean_degree_found = 0

            val_inputs = prepare_inputs(contents, num_neg_sampling=args.num_neg_sampling, dataset='valid', tc=time_cost)
            val_data_loader = DataLoader(val_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper,
                                         pin_memory=False, shuffle=True)

            val_running_loss = 0
            for batch_ndx, sample in enumerate(val_data_loader):
                model.eval()

                src_idx_l, rel_idx_l, target_idx_l, cut_time_l = sample.src_idx, sample.rel_idx, sample.target_idx, sample.ts
                num_query += len(src_idx_l)
                degree_batch = model.ngh_finder.get_temporal_degree(src_idx_l, cut_time_l)
                mean_degree += sum(degree_batch)

                model.set_init(src_idx_l, rel_idx_l, target_idx_l, cut_time_l, batch_ndx + 1, 0)
                query_src_emb, query_rel_emb, query_time_emb, attending_nodes, attending_node_attention, memorized_embedding = model.initialize()
                for step in range(args.DP_steps):
                    attending_nodes, attending_node_attention, memorized_embedding, _ = \
                        model.flow(attending_nodes, attending_node_attention, memorized_embedding, query_src_emb,
                                   query_rel_emb, query_time_emb)
                entity_att_score, entities = model.get_entity_attn_score(attending_node_attention, attending_nodes)

                one_hot_label = torch.from_numpy(
                    np.array([int(v == target_idx_l[eg_idx]) for eg_idx, v in entities], dtype=np.float32)).to(device)
                loss = torch.nn.BCELoss()(entity_att_score, one_hot_label)
                val_running_loss += loss.item()

                # _, indices = segment_topk(entity_att_score, entities[:, 0], 10, sorted=True)
                # for i, target in enumerate(target_idx_l):
                #     top10 = entities[indices[i]]
                #     hit_1 += target == top10[0, 1]
                #     hit_3 += target in top10[:3, 1]
                #     hit_10 += target in top10[:, 1]
                target_rank_l, found_mask, target_rank_fil_l, target_rank_fil_t_l = segment_rank_fil(entity_att_score,
                                                                                                     entities,
                                                                                                     target_idx_l, sp2o,
                                                                                                     src_idx_l,
                                                                                                     rel_idx_l,
                                                                                                     val_spt2o)
                # print(target_rank_l)
                mean_degree_found += sum(degree_batch[found_mask])
                hit_1 += np.sum(target_rank_l == 1)
                hit_3 += np.sum(target_rank_l <= 3)
                hit_10 += np.sum(target_rank_l <= 10)
                hit_1_fil += np.sum(target_rank_fil_l == 1)
                hit_3_fil += np.sum(target_rank_fil_l <= 3)
                hit_10_fil += np.sum(target_rank_fil_l <= 10)
                hit_1_fil_t += np.sum(target_rank_fil_t_l == 1)
                hit_3_fil_t += np.sum(target_rank_fil_t_l <= 3)
                hit_10_fil_t += np.sum(target_rank_fil_t_l <= 10)
                found_cnt += np.sum(found_mask)
                MR_total += np.sum(target_rank_l)
                MR_found += len(found_mask) and np.sum(
                    target_rank_l[found_mask])  # if no subgraph contains ground truch, MR_found = 0 for this batch
                MRR_total += np.sum(1 / target_rank_l)
                MRR_found += len(found_mask) and np.sum(
                    1 / target_rank_l[found_mask])  # if no subgraph contains ground truth, MRR_found = 0 for this batch
                MRR_total_fil += np.sum(1 / target_rank_fil_l)
                MRR_total_fil_t += np.sum(1 / target_rank_fil_t_l)
            print(
                "Filtered performance (time dependent): Hits@1: {}, Hits@3: {}, Hits@10: {}, MRR: {}".format(
                    hit_1_fil_t / num_query,
                    hit_3_fil_t / num_query,
                    hit_10_fil_t / num_query,
                    MRR_total_fil_t / num_query))
            print(
                "Filtered performance (time independent): Hits@1: {}, Hits@3: {}, Hits@10: {}, MRR: {}".format(
                    hit_1_fil / num_query,
                    hit_3_fil / num_query,
                    hit_10_fil / num_query,
                    MRR_total_fil / num_query))
            print(
                "Raw performance: Hits@1: {}, Hits@3: {}, Hits@10: {}, Hits@Inf: {}, MR: {}, MRR: {}, degree: {}".format(
                    hit_1 / num_query,
                    hit_3 / num_query,
                    hit_10 / num_query,
                    found_cnt / num_query,
                    MR_total / num_query,
                    MRR_total / num_query,
                    mean_degree / num_query))
            if found_cnt:
                print("Among Hits@Inf: Hits@1: {}, Hits@3: {}, Hits@10: {}, MR: {}, MRR: {}, degree: {}".format(
                    hit_1 / found_cnt,
                    hit_3 / found_cnt,
                    hit_10 / found_cnt,
                    MR_found / found_cnt,
                    MRR_found / found_cnt,
                    mean_degree_found / found_cnt))
            else:
                print('No subgraph found the ground truth!!')
            if args.sqlite:
                sqlite_conn = create_connection(os.path.join(save_dir, 'tKGR.db'))
                with sqlite_conn:
                    placeholders = ', '.join('?' * len(logging_col))
                    sql_logging = 'INSERT INTO logging({}) VALUES ({})'.format(', '.join(logging_col), placeholders)
                    logging_epoch = (
                        checkpoint_dir, epoch, running_loss, val_running_loss / (batch_ndx + 1), hit_1 / num_query,
                        hit_3 / num_query,
                        hit_10 / num_query, found_cnt / num_query, MRR_total / num_query, hit_1_fil_t / num_query,
                        hit_3_fil_t / num_query, hit_10_fil_t / num_query, MRR_total_fil_t / num_query)
                    cur = sqlite_conn.cursor()
                    cur.execute(sql_logging, logging_epoch)

print("finished Training")
#     os.umask(oldmask)
