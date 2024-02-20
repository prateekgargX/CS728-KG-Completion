import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

import argparse

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size,seq_len]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(DEVICE)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding

from torch_geometric.nn.kge import KGEModel


class TransformerKG(KGEModel):
    r"""The ComplEx model from the `"Complex Embeddings for Simple Link
    Prediction" <https://arxiv.org/abs/1606.06357>`_ paper.

    :class:`ComplEx` models relations as complex-valued bilinear mappings
    between head and tail entities using the Hermetian dot product.
    The entities and relations are embedded in different dimensional spaces,
    resulting in the scoring function:

    .. math::
        d(h, r, t) = Re(< \mathbf{e}_h,  \mathbf{e}_r, \mathbf{e}_t>)

    .. note::

        For an example of using the :class:`ComplEx` model, see
        `examples/kge_fb15k_237.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        kge_fb15k_237.py>`_.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
            the embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        sparse: bool = False,
        ):
        super().__init__(num_nodes, num_relations, hidden_channels, sparse)

        # special tokens introduced by us
        self.CLS_  = torch.tensor([num_nodes + num_relations + 0]).to(DEVICE)
        self.SEP1_ = torch.tensor([num_nodes + num_relations + 1]).to(DEVICE)
        self.SEP2_ = torch.tensor([num_nodes + num_relations + 2]).to(DEVICE)
        self.END_  = torch.tensor([num_nodes + num_relations + 3]).to(DEVICE) 


        #self.reset_parameters()
        self.vocab_size = num_nodes + num_relations + 4

        d_hid = 2048  # dimension of the feedforward network model in ``nn.TransformerEncoder``
        nlayers = 8  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
        nhead = 8  # number of heads in ``nn.MultiheadAttention``
        dropout = 0.2  # dropout probability
        self.bbmodel = TransformerModel(self.vocab_size, hidden_channels, nhead, d_hid, nlayers, dropout)
        
        # need a model to convert the output logits into a single score

        self.score_model = nn.Sequential(
                            nn.Linear(self.vocab_size, self.vocab_size//8),
                            nn.ReLU(),
                            nn.Linear(self.vocab_size//8,1)
                            )

    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        ) -> Tensor:

        batch_tokens = torch.stack([self.CLS_.expand_as(head_index), 
                                    head_index, 
                                    self.SEP1_.expand_as(head_index) ,
                                    rel_type + self.num_nodes,             # offset needed because single embed table
                                    self.SEP2_.expand_as(head_index), 
                                    tail_index, 
                                    self.END_.expand_as(head_index)], dim=1)
        
        output = self.bbmodel(batch_tokens) # batch,seq,vocabsize

        sscore = self.score_model(output[:,0,:]).squeeze() # pool only CLS part 
        return sscore

    def loss(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        ) -> Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))
        scores = torch.cat([pos_score, neg_score], dim=0)

        pos_target = torch.ones_like(pos_score)
        neg_target = torch.zeros_like(neg_score)
        target = torch.cat([pos_target, neg_target], dim=0)

        return F.binary_cross_entropy_with_logits(scores, target)


import os.path as osp

import torch

from torch_geometric.datasets import FB15k_237, WordNet18RR
from torch_geometric.nn import ComplEx
from torch_geometric.data import Data

from mutils import *

import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument('--num-epochs', type=int, default=500,required=True)
parser.add_argument('--val-every', type=int, default=25,required=False)
parser.add_argument('--dataset', type=str.upper, required=True, choices=['FB15K', 'WN18RR'])
args = parser.parse_args()

NUM_EPOCHS = args.num_epochs # Train for 500 Epochs
VAL_EVERY = args.val_every # Evaluate every 25 epochs
DATASET  = args.dataset

EMBED_DIM = 256

BATCH_SIZE_TRAIN = 1024
BATCH_SIZE_TEST  = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', f'{DATASET}')

if DATASET == 'FB15K':
    train_data = FB15k_237(PATH, split='train')[0].to(DEVICE)
    val_data = FB15k_237(PATH, split='val')[0].to(DEVICE)
    test_data = FB15k_237(PATH, split='test')[0].to(DEVICE)    
if DATASET == 'WN18RR':
    all_data = WordNet18RR(PATH)[0]

    train_data = Data(edge_index=all_data.edge_index[:,all_data.train_mask],
                    edge_type=all_data.edge_type[all_data.train_mask],
                    num_edge_types = all_data.num_edge_types,
                    num_nodes=all_data.num_nodes).to(DEVICE)
    val_data = Data(edge_index=all_data.edge_index[:,all_data.val_mask],
                    edge_type=all_data.edge_type[all_data.val_mask],
                    num_edge_types = all_data.num_edge_types,
                    num_nodes=all_data.num_nodes).to(DEVICE)
    test_data = Data(edge_index=all_data.edge_index[:,all_data.test_mask],
                    edge_type=all_data.edge_type[all_data.test_mask],
                    num_edge_types = all_data.num_edge_types,
                    num_nodes=all_data.num_nodes).to(DEVICE)

model = TransformerKG(
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=EMBED_DIM,
).to(DEVICE)

loader = model.loader(
    head_index=train_data.edge_index[0],
    rel_type=train_data.edge_type,
    tail_index=train_data.edge_index[1],
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
)
optimizer = optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6)

def train():
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples


@torch.no_grad()
def test_tail(data):
    model.eval()
    return model.test(model,
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=BATCH_SIZE_TEST,
        k=10,
    )

@torch.no_grad()
def test_head(data):
    model.eval()
    return test_h(model,
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=BATCH_SIZE_TEST,
        k=10,
    )

# rank, mrr, hits_at_10 = test(test_data)
# raise
for epoch in range(1, NUM_EPOCHS+1):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch % VAL_EVERY == 0: 
        rank, mrr, hits1, hits10 = test_tail(val_data)
        print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, '
              f'Val MRR: {mrr:.4f}, Val Hits@10: {hits10:.4f}')


print("Tail removed:")
rank, mrr, hits_at_1, hits_at_10 = test_tail(test_data)
mAp = calc_map_t(model,
         head_index=test_data.edge_index[0],
         rel_type=test_data.edge_type,
         tail_index=test_data.edge_index[1],
         batch_size=BATCH_SIZE_TEST)
print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
      f'Test Hits@10: {hits_at_10:.4f}',f'Test Hits@1: {hits_at_1:.4f}')
print(f'mAP: {mAp}')

print("Head removed:")
rank, mrr, hits_at_1, hits_at_10 = test_head(test_data)
mAp = calc_map_h(model,
         head_index=test_data.edge_index[0],
         rel_type=test_data.edge_type,
         tail_index=test_data.edge_index[1],
         batch_size=BATCH_SIZE_TEST)
print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
      f'Test Hits@10: {hits_at_10:.4f}',f'Test Hits@1: {hits_at_1:.4f}')
print(f'mAP: {mAp}')