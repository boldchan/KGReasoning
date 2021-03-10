import os
import sys
import argparse
import time
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

PackageDir = os.path.dirname(__file__)
sys.path.insert(1, PackageDir)

from utils import Data, NeighborFinder, Measure, save_config, get_git_version_short_hash, get_git_description_last_commit, load_checkpoint, new_checkpoint
from model import xERTE
from segment import *
from database_op import DBDriver
from train import reset_time_cost, str_time_cost, prepare_inputs, SimpleCustomBatch, collate_wrapper
def training(args):
    # check cuda
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.device) if args.device >= 0 else 'cpu'
    else:
        device = 'cpu'

    # profile time consumption
    time_cost = None
    if args.timer:
        time_cost = reset_time_cost()

    # init model and checkpoint folder
    start_time = time.time()
    struct_time = time.gmtime(start_time)
    epoch_command = args.epoch
    if args.load_checkpoint is None:
        checkpoint_dir, CHECKPOINT_PATH = new_checkpoint(save_dir, struct_time)
        contents = Data(dataset=args.dataset, add_reverse_relation=args.add_reverse)

        adj = contents.get_adj_dict()
        max_time = max(contents.data[:, 3])

        # construct NeighborFinder
        if 'yago' in args.dataset.lower():
            time_granularity = 1
        elif 'icews' in args.dataset.lower():
            time_granularity = 24
        else:
            raise ValueError
        nf = NeighborFinder(adj, sampling=args.sampling, max_time=max_time, num_entities=contents.num_entities,
                            weight_factor=args.weight_factor, time_granularity=time_granularity)
        # construct model
        model = xERTE(nf, contents.num_entities, contents.num_relations, args.emb_dim, DP_steps=args.DP_steps,
                       DP_num_edges=args.DP_num_edges, max_attended_edges=args.max_attended_edges,
                       node_score_aggregation=args.node_score_aggregation, ent_score_aggregation=args.ent_score_aggregation,
                       ratio_update=args.ratio_update, device=device, diac_embed=args.diac_embed, emb_static_ratio=args.emb_static_ratio,
                       update_prev_edges=not args.stop_update_prev_edges, use_time_embedding=not args.no_time_embedding,
                       attention_func = args.attention_func)
        # move a model to GPU before constructing an optimizer, http://pytorch.org/docs/master/optim.html
        model.to(device)
        model.entity_raw_embed.cpu()
        model.relation_raw_embed.cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        start_epoch = 0
        if not args.debug:
            print("Save checkpoints under {}".format(CHECKPOINT_PATH))
    else:
        raise ValueError('Do not use checkpoints')

    # save configuration to database and file system
    if not args.debug:
        dbDriver = DBDriver(useMongo=args.mongo, useSqlite=args.sqlite, MongoServerIP=local_config.MongoServer,
                            sqlite_dir=os.path.join(save_dir, 'tKGR.db'))
        git_hash = get_git_version_short_hash()
        git_comment = get_git_description_last_commit()
        dbDriver.log_task(args, checkpoint_dir, git_hash=git_hash, git_comment=git_comment,
                          device=local_config.AWS_device)
        save_config(args, CHECKPOINT_PATH)

    sp2o = contents.get_sp2o()
    val_spt2o = contents.get_spt2o('valid')
    train_inputs = prepare_inputs(contents, start_time=args.warm_start_time, tc=time_cost)
    val_inputs = prepare_inputs(contents, dataset='valid', tc=time_cost)
    analysis_data_loader = DataLoader(val_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper,
                                      pin_memory=False, shuffle=True)
    analysis_batch = next(iter(analysis_data_loader))

    for epoch in range(start_epoch, args.epoch):
        print("epoch: ", epoch)
        # load data
        train_inputs = prepare_inputs(contents, start_time=args.warm_start_time, tc=time_cost)
        train_data_loader = DataLoader(train_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper,
                                       pin_memory=False, shuffle=True)

        running_loss = 0.

        for batch_ndx, sample in tqdm(enumerate(train_data_loader)):
            if args.explainability_analysis and batch_ndx % 50 == 0:
                assert args.mongo
                mongodb_analysis_collection_name = 'analysis_' + checkpoint_dir
                src_idx_l, rel_idx_l, target_idx_l, cut_time_l = analysis_batch.src_idx, analysis_batch.rel_idx, analysis_batch.target_idx, analysis_batch.ts
                mongo_id = dbDriver.register_query_mongo(mongodb_analysis_collection_name, src_idx_l, rel_idx_l,
                                                         cut_time_l,
                                                         target_idx_l, vars(args), contents.id2entity,
                                                         contents.id2relation)
                model.eval()
                entity_att_score, entities, tracking = model(analysis_batch, analysis=True)
                target_rank_l, found_mask, target_rank_fil_l, target_rank_fil_t_l = segment_rank_fil(
                    entity_att_score,
                    entities,
                    target_idx_l,
                    sp2o,
                    val_spt2o,
                    src_idx_l,
                    rel_idx_l,
                    cut_time_l)
                for i in range(args.batch_size):
                    for step in range(args.DP_steps):
                        tracking[i][str(step)]["source_nodes(semantics)"] = [[contents.id2entity[n[1]], str(n[2])]
                                                                             for n
                                                                             in
                                                                             tracking[i][str(step)]["source_nodes"]]
                        tracking[i][str(step)]["sampled_edges(semantics)"] = [
                            [contents.id2entity[edge[1]], str(edge[2]),
                             contents.id2entity[edge[3]], str(edge[4]),
                             contents.id2relation[edge[5]]]
                            for edge in
                            tracking[i][str(step)]["sampled_edges"]]
                        tracking[i][str(step)]["selected_edges(semantics)"] = [
                            [contents.id2entity[edge[1]], str(edge[2]),
                             contents.id2entity[edge[3]], str(edge[4]),
                             contents.id2relation[edge[5]]]
                            for edge in
                            tracking[i][str(step)]["selected_edges"]]
                        tracking[i][str(step)]["new_sampled_nodes(semantics)"] = [
                            [contents.id2entity[n[1]], str(n[2])]
                            for
                            n in tracking[i][str(step)][
                                "new_sampled_nodes"]]
                        tracking[i][str(step)]["new_source_nodes(semantics)"] = [
                            [contents.id2entity[n[1]], str(n[2])]
                            for n
                            in
                            tracking[i][str(step)][
                                "new_source_nodes"]]
                    tracking[i]['entity_candidate(semantics)'] = [contents.id2entity[ent] for ent in
                                                                  tracking[i]['entity_candidate']]
                    tracking[i]['epoch'] = epoch
                    tracking[i]['batch_idx'] = batch_ndx
                    dbDriver.mongodb[mongodb_analysis_collection_name].update_one({"_id": mongo_id[i]},
                                                                                  {"$set": tracking[i]})
            optimizer.zero_grad()
            model.zero_grad()
            model.train()

            entity_att_score, entities = model(sample)
            target_idx_l = sample.target_idx

            loss = model.loss(entity_att_score, entities, target_idx_l, args.batch_size,
                              args.gradient_iters_per_update, args.loss_fn)
            if args.timer:
                t_start = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if args.timer:
                time_cost['grad']['comp'] += time.time() - t_start

            if args.timer:
                t_start = time.time()
            if (batch_ndx + 1) % args.gradient_iters_per_update == 0:
                optimizer.step()
                model.zero_grad()
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
            'args': args
        }, os.path.join(CHECKPOINT_PATH, 'checkpoint_{}.pt'.format(epoch)))

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

    val_data_loader = DataLoader(val_inputs, batch_size=args.batch_size, collate_fn=collate_wrapper,
                                 pin_memory=False, shuffle=True)

    val_running_loss = 0
    for batch_ndx, sample in enumerate(val_data_loader):
        model.eval()

        src_idx_l, rel_idx_l, target_idx_l, cut_time_l = sample.src_idx, sample.rel_idx, sample.target_idx, sample.ts
        num_query += len(src_idx_l)
        degree_batch = model.ngh_finder.get_temporal_degree(src_idx_l, cut_time_l)
        mean_degree += sum(degree_batch)

        entity_att_score, entities = model(sample)

        loss = model.loss(entity_att_score, entities, target_idx_l, args.batch_size,
                          args.gradient_iters_per_update, args.loss_fn)

        val_running_loss += loss.item()

        # _, indices = segment_topk(entity_att_score, entities[:, 0], 10, sorted=True)
        # for i, target in enumerate(target_idx_l):
        #     top10 = entities[indices[i]]
        #     hit_1 += target == top10[0, 1]
        #     hit_3 += target in top10[:3, 1]
        #     hit_10 += target in top10[:, 1]
        target_rank_l, found_mask, target_rank_fil_l, target_rank_fil_t_l = segment_rank_fil(
            entity_att_score,
            entities,
            target_idx_l,
            sp2o,
            val_spt2o,
            src_idx_l,
            rel_idx_l,
            cut_time_l)
        # print(target_rank_l)
        mean_degree_found += sum(degree_batch[found_mask])
        hit_1 += np.sum(target_rank_l == 1)
        hit_3 += np.sum(target_rank_l <= 3)
        hit_10 += np.sum(target_rank_l <= 10)
        hit_1_fil += np.sum(target_rank_fil_l <= 1)  # unique entity with largest node score
        hit_3_fil += np.sum(target_rank_fil_l <= 3)
        hit_10_fil += np.sum(target_rank_fil_l <= 10)
        hit_1_fil_t += np.sum(target_rank_fil_t_l <= 1)  # unique entity with largest node score
        hit_3_fil_t += np.sum(target_rank_fil_t_l <= 3)
        hit_10_fil_t += np.sum(target_rank_fil_t_l <= 10)
        found_cnt += np.sum(found_mask)
        MR_total += np.sum(target_rank_l)
        MR_found += len(found_mask) and np.sum(
            target_rank_l[found_mask])  # if no subgraph contains ground truch, MR_found = 0 for this batch
        MRR_total += np.sum(1 / target_rank_l)
        MRR_found += len(found_mask) and np.sum(
            1 / target_rank_l[
                found_mask])  # if no subgraph contains ground truth, MRR_found = 0 for this batch
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

    performance_key = ['training_loss', 'validation_loss', 'HITS_1_raw', 'HITS_3_raw', 'HITS_10_raw',
                       'HITS_INF', 'MRR_raw', 'HITS_1_found', 'HITS_3_found', 'HITS_10_found', 'MRR_found']
    performance = [running_loss, val_running_loss / (batch_ndx + 1), hit_1 / num_query,
                   hit_3 / num_query,
                   hit_10 / num_query, found_cnt / num_query, MRR_total / num_query,
                   hit_1_fil_t / num_query,
                   hit_3_fil_t / num_query, hit_10_fil_t / num_query, MRR_total_fil_t / num_query]
    performance_dict = {k: float(v) for k, v in zip(performance_key, performance)}

    dbDriver.log_evaluation(checkpoint_dir, epoch, performance_dict)
    return  performance[2], checkpoint_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dim', type=int, default=[256, 128, 64, 32], nargs='+', help='dimension of embedding for node, realtion and time')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_attended_edges', type=int, default=60, help='max number of edges after pruning')
    parser.add_argument('--ratio_update', type=float, default=0, help='ratio_update: when update node representation: '
                                                                      'ratio * self representation + (1 - ratio) * neighbors, '
                                                                      'if ratio==0, GCN style, ratio==1, no node representation update')

    parser.add_argument('--DP_num_edges', type=int, default=15, help='number of edges at each sampling')
    parser.add_argument('--DP_steps', type=int, default=3, help='number of DP steps')
    parser.add_argument('--dataset', type=str, default=None, help='specify data set')
    parser.add_argument('--whole_or_seen', type=str, default='whole', choices=['whole', 'seen', 'unseen'],
                        help='test on the whole set or only on seen entities.')
    parser.add_argument('--warm_start_time', type=int, default=48, help="training data start from what timestamp")
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--device', type=int, default=-1, help='-1: cpu, >=0, cuda device')
    parser.add_argument('--sampling', type=int, default=3,
                        help='strategy to sample neighbors, 0: uniform, 1: first num_neighbors, 2: last num_neighbors, 3: time-difference weighted')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--weight_factor', type=float, default=2, help='sampling 3, scale weight')
    parser.add_argument('--node_score_aggregation', type=str, default='sum', choices=['sum', 'mean', 'max'])
    parser.add_argument('--ent_score_aggregation', type=str, default='sum', choices=['sum', 'mean'])
    parser.add_argument('--emb_static_ratio', type=float, default=1,
                        help='ratio of static embedding to time(temporal) embeddings')
    parser.add_argument('--add_reverse', action='store_true', default=True, help='add reverse relation into data set')
    parser.add_argument('--attention_func', type=str, default='G3', help='choice of attention functions')
    parser.add_argument('--loss_fn', type=str, default='BCE', choices=['BCE', 'CE'])
    parser.add_argument('--stop_update_prev_edges', action='store_true', default=False,
                        help='stop updating node representation along previous selected edges')
    parser.add_argument('--no_time_embedding', action='store_true', default=False,
                        help='set to stop use time embedding')
    parser.add_argument('--explainability_analysis', action='store_true', default=None,
                        help='set to return middle output for explainability analysis')
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--sqlite', action='store_true', default=None, help='save information to sqlite')
    parser.add_argument('--mongo', action='store_true', default=None, help='save information to mongoDB')

    parser.add_argument('--gradient_iters_per_update', type=int, default=1,
                        help='gradient accumulation, update parameters every N iterations, default 1. set when GPU memo is small')
    parser.add_argument('--timer', action='store_true', default=None,
                        help='set to profile time consumption for some func')
    parser.add_argument('--debug', action='store_true', default=None,
                        help='in debug mode, checkpoint will not be saved')
    parser.add_argument('--diac_embed', action='store_true',
                        help='use entity-specific frequency and phase of time embeddings')
    args = parser.parse_args()

    if not args.debug:
        import local_config

        save_dir = local_config.save_dir
    else:
        save_dir = ''

    best_val = 0
    best_epoch = 0
    best_checkpoint_dir = None
    for ratioupdate in [0.25, 0.75]:
            for dims in [[512, 256, 128, 64], [256, 128, 64, 32]]:
                for DP_steps in [2, 3]:
                    if DP_steps == 2:
                        args.emb_dim = [256, 128, 64]
                    elif DP_steps == 3:
                        args.emb_dim = [256, 128, 64, 32]
                    else:
                        raise NotImplemented
                    args.emb_dim = dims
                    args.ratio_update = ratioupdate
                    args.DP_steps = DP_steps
                    val_hits1, val_checkpoint_dir = training(args)
                    if best_val < val_hits1:
                        best_val = val_hits1
                        best_checkpoint_dir = val_checkpoint_dir

    dbDriver.close()
    print("finished hyperparameter tuning")
    print("start evaluation on test set")
    os.system("python eval.py --load_checkpoint {}/checkpoint_{}.pt --whole_or_seen {} --device {} --mongo".format(
        best_checkpoint_dir, args.epoch, args.whole_or_seen, args.device))


# python HyperParaTuning.py  --dataset ICEWS18_forecasting  --device 0 --mongo