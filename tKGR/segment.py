import time

import numpy as np
import torch

def _segment_id2sparse_block_diag_matrix_coordinate(segment_ids):
    """
    segment_ids is a ascending 1d numpy array, dtype int, e.g. [0,0,0,1,2,2,3, ...]
    we want to create a sparse block digonal matrix from segment_ids,
    i-th block (square) has a shape (number_of_i_in_segment_ids, number_of_i_in_segment_ids)
    and each block is filled with 1
    e.g. [0,0,0,1,2,2] -->
    [[1,1,1,0,0,0],
     [1,1,1,0,0,0],
     [1,1,1,0,0,0],
     [0,0,0,1,0,0],
     [0,0,0,0,1,1],
     [0,0,0,0,1,1]]
    Attention!: But we don't return the matrix, we return the index of nonzero in this matrix
    in the form of a numpy array of shape 2 x N, first row is row index, second row is col index
    """
    mask = segment_ids[:-1] == segment_ids[1:]
    segment_start = np.concatenate([np.array([0]),
                                    np.arange(1, len(segment_ids))[mask],
                                    np.array([len(segment_ids)])])
    segment_len = np.diff(segment_start)

    row_idx = []
    col_idx = []
    shift = 0
    for i, slen in enumerate(segment_len):
        shift += i and segment_len[i - 1]
        col_idx.append(np.tile(np.arange(slen), slen) + shift)
        row_idx.append(np.repeat(np.arange(slen), slen) + shift)
    col_idx = np.concatenate(col_idx)
    row_idx = np.concatenate(row_idx)
    return np.stack([row_idx, col_idx], axis=0)


def segment_softmax_op(logits, segment_ids, tc=None):
    """
    logits is a 1d tensor of attention score (refer to DPMPN paper),
    i-th  node has attention score logits[i] which is in the subgraph developed for the query segment_ids[i]
    This function try to calculate the softmax of the nodes in the same subgraph

    :param logits: 1d Tensor
    :param segment_ids: id numpy.array eg_idx, sorted
    :return:
    softmax for logtis with same segment_id
    """
    device = logits.get_device()
    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device))

    len_logits = len(segment_ids)
    if tc:
        t_start = time.time()
    sparse_index_np = _segment_id2sparse_block_diag_matrix_coordinate(segment_ids)
    if tc:
        tc['model']['DP_attn_softmax_trans_matrix'] = time.time() - t_start
    sparse_index = torch.LongTensor(sparse_index_np)
    sparse_value = torch.ones(sparse_index_np.shape[1], dtype=torch.float)
    trans_matrix_sparse_th = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                      torch.Size([len_logits, len_logits])).to(device)
    softmax_den = torch.squeeze(torch.sparse.mm(trans_matrix_sparse_th, torch.exp(logits).unsqueeze(1)))
    logits_segment_softmax = torch.exp(logits) / softmax_den
    # logits_segment_softmax = logits.clone()
    # for segment_id in sorted(set(segment_ids)):
    #     # segment_mask = segment_ids==segment_id # somehow no grad is generated
    #     segment_idx = np.where(segment_ids == segment_id)[0]
    #     logits_max = torch.max(logits[segment_idx])
    #     logits_diff = logits[segment_idx] - logits_max
    #     logits_exp = torch.exp(logits_diff)
    #     logits_expsum = torch.sum(logits_exp)
    #     logits_norm = logits_exp / logits_expsum
    #     logits_segment_softmax[segment_idx] = logits_norm
    return logits_segment_softmax


def segment_sum(logits, segment_ids, keep_length=True):
    """

    :param logits:
    :param segment_ids:
    :param keep_length: if True, return a Tensor with the same length as logits
    out[i] is the sum of segments of segment_ids[i]
    else, return a Tensor with the length of segment_ids
    :return:
    """
    device = logits.get_device()
    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device))

    logits_len = len(segment_ids)
    num_segments = max(segment_ids) + 1

    # calculate summation of exponential of logits value for each group
    sparse_index = torch.LongTensor(np.stack([segment_ids, np.arange(logits_len)]))
    sparse_value = torch.ones(logits_len, dtype=torch.float)
    trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                   torch.Size([num_segments, logits_len])).to(device)
    seg_sum = torch.sparse.mm(trans_matrix_sparse, logits.unsqueeze(1))
    if not keep_length:
        return seg_sum

    # repeat softmax denominator to have the same length as logits
    sparse_index2 = torch.LongTensor(np.stack([np.arange(logits_len), segment_ids]))
    sparse_value2 = torch.ones(logits_len, dtype=torch.float)
    trans_matrix_sparse2 = torch.sparse.FloatTensor(sparse_index2, sparse_value2,
                                                    torch.Size([logits_len, num_segments])).to(device)
    seg_sum_repeat = torch.sparse.mm(trans_matrix_sparse2, seg_sum)
    return torch.squeeze(seg_sum_repeat)


def segment_max(logits, segment_ids, keep_length=True):
    """

    :param logits:
    :param segment_ids:
    :param keep_length:
    if True, return a Tensor with the same length as logits
    out[i] is the sum of segments of segment_ids[i]
    else, return a Tensor with the length of segment_ids
    :return:
    1d Tensor
    """
    n_logits = len(segment_ids)
    mask = segment_ids[1:] != segment_ids[:-1]
    seg_head_ids = np.concatenate([np.array([0]),
                                   np.arange(1, n_logits)[mask],
                                   np.array([n_logits])]).astype(np.int64)
    if keep_length:
        seg_max_ind = torch.cat([(torch.argmax(logits[torch.arange(head, tail).to(torch.int64)]) + torch.tensor([head]).to(torch.int64)).repeat(tail - head) for head, tail in zip(seg_head_ids[:-1], seg_head_ids[1:])])
    else:
        seg_max_ind = torch.cat([torch.argmax(logits[torch.arange(head, tail).to(torch.int64)]) + torch.tensor([head]).to(torch.int64) for head, tail in zip(seg_head_ids[:-1], seg_head_ids[1:])])
    return logits[seg_max_ind]


def segment_softmax_op_v2(logits, segment_ids, tc=None):
    device = logits.get_device()
    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device))

    if tc:
        t_start = time.time()

    logits_len = len(segment_ids)
    num_segments = max(segment_ids) + 1
    # numerical stable softmax
    logits = logits - segment_max(logits, segment_ids, keep_length=True)
    logits_exp = torch.exp(logits).unsqueeze(1)  # e^{logit} N x 1

    # calculate summation of exponential of logits value for each group
    sparse_index = torch.LongTensor(np.stack([segment_ids, np.arange(logits_len)]))
    sparse_value = torch.ones(logits_len, dtype=torch.float)
    trans_matrix_sparse = torch.sparse.FloatTensor(sparse_index, sparse_value,
                                                   torch.Size([num_segments, logits_len])).to(device)
    softmax_den = torch.sparse.mm(trans_matrix_sparse, logits_exp)

    # repeat softmax denominator to have the same length as logits
    sparse_index2 = torch.LongTensor(np.stack([np.arange(logits_len), segment_ids]))
    sparse_value2 = torch.ones(logits_len, dtype=torch.float)
    trans_matrix_sparse2 = torch.sparse.FloatTensor(sparse_index2, sparse_value2,
                                                    torch.Size([logits_len, num_segments])).to(device)
    softmax_den_repeat = torch.sparse.mm(trans_matrix_sparse2, softmax_den)

    out = torch.squeeze(logits_exp / softmax_den_repeat)
    if tc:
        tc['model']['DP_attn_softmax_v2'] += time.time() - t_start
    return out