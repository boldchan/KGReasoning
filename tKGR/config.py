import argparse


def get_ICEWS14_config(parser):
    parser.add_argument('--dataset', type=str, default='ICEWS14_forecasting', help='specify data set')
    parser.add_argument('--num_neg_sampling', type=int, default=5,
                        help="number of negative sampling of objects for each event")
    parser.add_argument('--num_layers', type=int, default=2, help='number of TGAN layers')
    parser.add_argument('--warm_start_time', type=int, default=48, help="training data start from what timestamp")
    parser.add_argument('--node_feat_dim', type=int, default=100, help='dimension of embedding for node')
    parser.add_argument('--edge_feat_dim', type=int, default=100, help='dimension of embedding for edge')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_neighbors', type=int, default=20,
                        help='how many neighbors to aggregate information from, '
                             'check paper Inductive Representation Learning '
                             'for Temporal Graph for detail')
    parser.add_argument('--sampling', type=int, default=1, help="neighborhood sampling strategy, 0: uniform, 1: first num_neighbors, 2: last num_neighbors")
    parser.add_argument('--device', type=int, default=-1, help='-1: cpu, >=0, cuda device')
    parser.add_argument('--add_reverse', action='store_true', default=False)
    return parser


def get_ICEWS18_config(parser):
    parser.add_argument('--dataset', type=str, default='ICEWS18_forecasting', help='specify data set')
    parser.add_argument('--num_neg_sampling', type=int, default=5,
                        help="number of negative sampling of objects for each event")
    parser.add_argument('--num_layers', type=int, default=2, help='number of TGAN layers')
    parser.add_argument('--warm_start_time', type=int, default=48, help="training data start from what timestamp")
    parser.add_argument('--node_feat_dim', type=int, default=100, help='dimension of embedding for node')
    parser.add_argument('--edge_feat_dim', type=int, default=100, help='dimension of embedding for edge')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_neighbors', type=int, default=20,
                        help='how many neighbors to aggregate information from, '
                             'check paper Inductive Representation Learning '
                             'for Temporal Graph for detail')
    parser.add_argument('--uniform', action='store_true', help="uniformly sample num_neighbors neighbors")
    parser.add_argument('--device', type=int, default=-1, help='-1: cpu, >=0, cuda device')
    parser.add_argument('--add_reverse', action='store_true', default=True)
    return parser


def get_default_config(name):
    parser = argparse.ArgumentParser()
    if name == 'ICEWS14_forecasting':
        return get_ICEWS14_config(parser)
    elif name == 'ICEWS18_forecasting':
        return get_ICEWS18_config(parser)
    else:
        raise ValueError("Invalid name")
