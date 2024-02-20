from typing import Tuple

import torch
from torch import Tensor
from tqdm import tqdm

@torch.no_grad()
def test_t(
    model,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    batch_size: int,
    k: int = 10,
    log: bool = True,
) -> Tuple[float, float, float]:
    r"""Evaluates the model quality by computing Mean Rank, MRR and
    Hits@:math:`k` across all possible tail entities.

    Args:
        head_index (torch.Tensor): The head indices.
        rel_type (torch.Tensor): The relation type.
        tail_index (torch.Tensor): The tail indices.
        batch_size (int): The batch size to use for evaluating.
        k (int, optional): The :math:`k` in Hits @ :math:`k`.
            (default: :obj:`10`)
        log (bool, optional): If set to :obj:`False`, will not print a
            progress bar to the console. (default: :obj:`True`)
    """
    arange = range(head_index.numel())
    arange = tqdm(arange) if log else arange

    mean_ranks, reciprocal_ranks, hits_at_1, hits_at_10 = [], [], [], []
    for i in arange:
        h, r, t = head_index[i], rel_type[i], tail_index[i]

        scores = []
        tail_indices = torch.arange(model.num_nodes, device=t.device)
        for ts in tail_indices.split(batch_size):
            scores.append(model(h.expand_as(ts), r.expand_as(ts), ts))
        rank = int((torch.cat(scores).argsort(
            descending=True) == t).nonzero().view(-1))
        mean_ranks.append(rank)
        reciprocal_ranks.append(1 / (rank + 1))
        hits_at_1.append(rank < 1)
        hits_at_10.append(rank < 10)

    mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
    mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
    hits_at_1 = int(torch.tensor(hits_at_1).sum()) / len(hits_at_1)
    hits_at_10 = int(torch.tensor(hits_at_10).sum()) / len(hits_at_10)

    return mean_rank, mrr, hits_at_1, hits_at_10

# code modifed from test function in KGEModel class
@torch.no_grad()
def test_h(
    model,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    batch_size: int,
    k: int = 10,
    log: bool = True,
) -> Tuple[float, float, float]:
    r"""Evaluates the model quality by computing Mean Rank, MRR and
    Hits@:math:`k` across all possible head entities.

    Args:
        head_index (torch.Tensor): The head indices.
        rel_type (torch.Tensor): The relation type.
        tail_index (torch.Tensor): The tail indices.
        batch_size (int): The batch size to use for evaluating.
        k (int, optional): The :math:`k` in Hits @ :math:`k`.
            (default: :obj:`10`)
        log (bool, optional): If set to :obj:`False`, will not print a
            progress bar to the console. (default: :obj:`True`)
    """
    arange = range(head_index.numel())
    arange = tqdm(arange) if log else arange

    mean_ranks, reciprocal_ranks, hits_at_1, hits_at_10 = [], [], [], []
    for i in arange:
        h, r, t = head_index[i], rel_type[i], tail_index[i]

        scores = []
        head_indices = torch.arange(model.num_nodes, device=h.device)
        for hs in head_indices.split(batch_size):
            scores.append(model(hs, r.expand_as(hs), t.expand_as(hs)))
        rank = int((torch.cat(scores).argsort(
            descending=True) == h).nonzero().view(-1))
        mean_ranks.append(rank)
        reciprocal_ranks.append(1 / (rank + 1))
        hits_at_1.append(rank < 1)
        hits_at_10.append(rank < 10)

    mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
    mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
    hits_at_1 = int(torch.tensor(hits_at_1).sum()) / len(hits_at_1)
    hits_at_10 = int(torch.tensor(hits_at_10).sum()) / len(hits_at_10)

    return mean_rank, mrr, hits_at_1, hits_at_10

@torch.no_grad()
def calc_map_t(
    model,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    batch_size: int,
) -> Tuple[float, float, float]:
    
    queries = torch.stack((head_index, rel_type), dim=1).unique(dim=0)
    num_queries = queries.shape[0]
    output  = torch.zeros((num_queries, model.num_nodes), device=head_index.device)

    Y = torch.zeros((num_queries, model.num_nodes), device=head_index.device)

    stacked_hr = torch.stack((head_index, rel_type), dim=1)

    # prepare Y
    for i in range(head_index.numel()):
        h, r, t = head_index[i], rel_type[i], tail_index[i]
        Y[((stacked_hr[i] == queries)[:,0]),t] = 1

    #  prepare output
    for i in range(num_queries):
        h,r = queries[i,0], queries[i,1]
        scores = []
        tail_indices = torch.arange(model.num_nodes, device=t.device)
        for ts in tail_indices.split(batch_size):
            scores.append(model(h.expand_as(ts), r.expand_as(ts), ts))
        scores = torch.cat(scores)
        output[i] = scores

    mAP = []
    # mAP computation
    ranked_output_idx = torch.argsort(output, descending=True)
    ranked_output = torch.gather(Y, 1, ranked_output_idx)
    cumulative_sum = (torch.cumsum(ranked_output, dim=1) * ranked_output)   # [0, 1, 2, 0, 3, 4]
    div_factor = torch.arange(model.num_nodes, device=head_index.device)            
    cumulative_sum = cumulative_sum.float() / (div_factor + 1)              
    cumulative_sum = cumulative_sum / torch.sum(ranked_output, dim=1).unsqueeze(1)
    mAP.append(cumulative_sum.sum().item())

    return mAP[0]/num_queries

@torch.no_grad()
def calc_map_h(
    model,
    head_index: Tensor,
    rel_type: Tensor,
    tail_index: Tensor,
    batch_size: int,
) -> Tuple[float, float, float]:
    
    queries = torch.stack((tail_index, rel_type), dim=1).unique(dim=0)
    num_queries = queries.shape[0]
    output  = torch.zeros((num_queries, model.num_nodes), device=head_index.device)

    Y = torch.zeros((num_queries, model.num_nodes), device=head_index.device)

    stacked_hr = torch.stack((tail_index, rel_type), dim=1)

    # prepare Y
    for i in range(head_index.numel()):
        h, r, t = head_index[i], rel_type[i], tail_index[i]
        Y[((stacked_hr[i] == queries)[:,0]),h] = 1

    #  prepare output
    for i in range(num_queries):
        t,r = queries[i,0], queries[i,1]
        scores = []
        head_indices = torch.arange(model.num_nodes, device=t.device)
        for hs in head_indices.split(batch_size):
            scores.append(model(hs, r.expand_as(hs), t.expand_as(hs)))
        scores = torch.cat(scores)
        output[i] = scores

    mAP = []
    # mAP computation
    ranked_output_idx = torch.argsort(output, descending=True)
    ranked_output = torch.gather(Y, 1, ranked_output_idx)
    cumulative_sum = (torch.cumsum(ranked_output, dim=1) * ranked_output)   # [0, 1, 2, 0, 3, 4]
    div_factor = torch.arange(model.num_nodes, device=head_index.device)            
    cumulative_sum = cumulative_sum.float() / (div_factor + 1)              
    cumulative_sum = cumulative_sum / torch.sum(ranked_output, dim=1).unsqueeze(1)
    mAP.append(cumulative_sum.sum().item())

    return mAP[0]/num_queries



# @torch.no_grad()
# def tail_test(
#     model,
#     head_index: Tensor,
#     rel_type: Tensor,
#     tail_index: Tensor,
#     batch_size: int,
#     k: int = 10,
#     log: bool = True,
# ) -> Tuple[float, float, float]:
    
#     queries = torch.stack((head_index, rel_type), dim=1).unique(dim=0)
#     num_queries = queries.shape[0]
#     output  = torch.zeros((num_queries, model.num_nodes), device=head_index.device)

#     Y = torch.zeros((num_queries, model.num_nodes), device=head_index.device)
#     stacked_hr = torch.stack((head_index, rel_type), dim=1)

#     # prepare Y
#     for i in range(head_index.numel()):
#         h, r, t = head_index[i], rel_type[i], tail_index[i]
#         Y[((stacked_hr[i] == queries)[:,0]),t] = 1

#     arange = range(num_queries)
#     arange = tqdm(arange) if log else arange

#     mean_ranks, reciprocal_ranks, hits_at_k = [], [], []

#     #  prepare output
#     for i in arange:
#         h,r = queries[i,0], queries[i,1]
#         scores = []
#         tail_indices = torch.arange(model.num_nodes, device=t.device)
#         for ts in tail_indices.split(batch_size):
#             scores.append(model(h.expand_as(ts), r.expand_as(ts), ts))
#         scores = torch.cat(scores)
#         output[i] = scores
#         rank = int((scores.argsort(
#             descending=True) == t).nonzero().view(-1))
#         mean_ranks.append(rank)
#         reciprocal_ranks.append(1 / (rank + 1))
#         hits_at_k.append(rank < k)
        

#     mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
#     mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
#     hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)


#     mAP = []
#     # mAP computation
#     ranked_output_idx = torch.argsort(output, descending=True)
#     ranked_output = torch.gather(Y, 1, ranked_output_idx)
#     cumulative_sum = (torch.cumsum(ranked_output, dim=1) * ranked_output)   # [0, 1, 2, 0, 3, 4]
#     div_factor = torch.arange(model.num_nodes, device=head_index.device)            
#     cumulative_sum = cumulative_sum.float() / (div_factor + 1)              
#     cumulative_sum = cumulative_sum / torch.sum(ranked_output, dim=1).unsqueeze(1)
#     mAP.append(cumulative_sum.sum().item())

#     return mAP[0]/num_queries
