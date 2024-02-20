import argparse
import os.path as osp

import torch
import torch.optim as optim

from torch_geometric.datasets import FB15k_237, WordNet18RR
from torch_geometric.nn import ComplEx
from torch_geometric.data import Data

# local imports
from mutils import *

parser = argparse.ArgumentParser()
parser.add_argument('--num-epochs', type=int, default=500,required=True)
parser.add_argument('--val-every', type=int, default=25,required=False)
parser.add_argument('--dataset', type=str.upper, required=True, choices=['FB15K', 'WN18RR'])
args = parser.parse_args()

NUM_EPOCHS = args.num_epochs # Train for 500 Epochs
VAL_EVERY = args.val_every # Evaluate every 25 epochs
DATASET  = args.dataset

EMBED_DIM = 50

BATCH_SIZE_TRAIN = 1000
BATCH_SIZE_TEST  = 20000
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

model = ComplEx(
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