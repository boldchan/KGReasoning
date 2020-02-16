import numpy as np
import torch

# attention layer based on GAT paper, attention score is calculated based on weighted summation of node features
# check how logits are calculated below
def attn_head(node_embeddings, out_size, adj_mat, activation, num_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    if in_drop != 0:
        node_embeddings = torch.nn.Dropout(p=in_drop)(node_embeddings)

    # feedforward
    hidden_features = torch.nn.Conv1d(num_nodes, out_size, 1, bias=False)(node_embeddings)

    # self-attention coefficients
    # two different weights or shared weights?
    f_1 = torch.nn.Conv1d(num_nodes, 1, 1)(hidden_features)
    f_2 = torch.nn.Conv1d(num_nodes, 1, 1)(hidden_features)

    f_1 = torch.reshape(f_1, (num_nodes, 1))
    f_2 = torch.reshape(f_2, (num_nodes, 1))

    f_1 = adj_mat * f_1
    f_2 = adj_mat * torch.t(f_2)

    logits = f_1+f_2 # look here!
    lrelu = torch.nn.LeakyReLU()(logits)
    coefs = torch.nn.Softmax()(lrelu)

    if coef_drop != 0.0:
        coefs = torch.nn.Dropout(p=coef_drop)(coefs)

    # why dropout again?
    if in_drop != 0.0:
        hidden_features = torch.nn.Dropout(p=in_drop)(hidden_features)

    vals = torch.matmul(coefs, hidden_features)

    if residual:
        if node_embeddings.shape[-1] != vals.shape[-1]:
            vals = vals + torch.nn.Conv1d(num_nodes, vals.shape[-1],1)(node_embeddings)
        else:
            vals += node_embeddings

    return activation()(vals)

# attention layer based on dot product, see formula 5 in Dynamically pruned message passing network
def attn_head_dot_product(node_embeddings, out_size, adj_mat, activation, num_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    if in_drop != 0:
        node_embeddings = torch.nn.Dropout(p=in_drop)(node_embeddings)

    # feedforward
    hidden_features = torch.nn.Conv1d(num_nodes, out_size, 1, bias=False)(node_embeddings)

    # self-attention coefficients
    # two different weights or shared weights?
    f_1 = torch.nn.Conv1d(num_nodes, 1, 1)(hidden_features)
    f_2 = torch.nn.Conv1d(num_nodes, 1, 1)(hidden_features)

    f_1 = torch.reshape(f_1, (num_nodes, 1))
    f_2 = torch.reshape(f_2, (num_nodes, 1))

    f_1 = adj_mat * f_1
    f_2 = adj_mat * torch.t(f_2)

    attention_weights = torch.autograd.Variable(torch.randn(out_size, out_size).type(torch.FloatTensor),
                                                requires_grad=True)
    logits = torch.matmul(torch.matmul(f_1, attention_weights), torch.t(f_2))  # look here!
    lrelu = torch.nn.LeakyReLU()(logits)
    coefs = torch.nn.Softmax()(lrelu)

    if coef_drop != 0.0:
        coefs = torch.nn.Dropout(p=coef_drop)(coefs)

    # why dropout again?
    if in_drop != 0.0:
        hidden_features = torch.nn.Dropout(p=in_drop)(hidden_features)

    vals = torch.matmul(coefs, hidden_features)

    if residual:
        if node_embeddings.shape[-1] != vals.shape[-1]:
            vals = vals + torch.nn.Conv1d(num_nodes, vals.shape[-1], 1)(node_embeddings)
        else:
            vals += node_embeddings

    return activation()(vals)


