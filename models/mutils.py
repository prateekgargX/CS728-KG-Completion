from typing import Tuple

import torch
from torch import Tensor
from tqdm import tqdm


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

    mean_ranks, reciprocal_ranks, hits_at_k = [], [], []
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
        hits_at_k.append(rank < k)

    mean_rank = float(torch.tensor(mean_ranks, dtype=torch.float).mean())
    mrr = float(torch.tensor(reciprocal_ranks, dtype=torch.float).mean())
    hits_at_k = int(torch.tensor(hits_at_k).sum()) / len(hits_at_k)

    return mean_rank, mrr, hits_at_k