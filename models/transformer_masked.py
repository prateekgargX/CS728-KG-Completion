import argparse
import os.path as osp
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import FB15k_237, WordNet18RR
from torch_geometric.data import Data

from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500,required=True)
parser.add_argument('--dataset', type=str.upper, required=True, choices=['FB15K', 'WN18RR'])
parser.add_argument('--device', type=int, required=True)
args = parser.parse_args()

dataset = args.dataset
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

class KGDataset:
    def __init__(self, dataset):

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', f'{dataset}')

        if dataset == 'FB15K':
            train_data = FB15k_237(path, split='train')[0].to(device)
            val_data = FB15k_237(path, split='val')[0].to(device)
            test_data = FB15k_237(path, split='test')[0].to(device)
        elif dataset == 'WN18RR':
            all_data = WordNet18RR(path)[0]
            train_data = Data(
                edge_index=all_data.edge_index[:,all_data.train_mask],
                edge_type=all_data.edge_type[all_data.train_mask],
                num_edge_types = all_data.num_edge_types,
                num_nodes=all_data.num_nodes
            ).to(device)
            val_data = Data(
                edge_index=all_data.edge_index[:,all_data.val_mask],
                edge_type=all_data.edge_type[all_data.val_mask],
                num_edge_types = all_data.num_edge_types,
                num_nodes=all_data.num_nodes
            ).to(device)
            test_data = Data(
                edge_index=all_data.edge_index[:,all_data.test_mask],
                edge_type=all_data.edge_type[all_data.test_mask],
                num_edge_types = all_data.num_edge_types,
                num_nodes=all_data.num_nodes
            ).to(device)

        self.num_nodes = train_data.num_nodes
        self.num_edge_types = len(torch.unique(train_data.edge_type))  
        # hopefully train_data has at least one of each type ..  

        # need to get token identifiers for CLS, SEP1, SEP2, MASK, END
        # taken ids 0 - num_nodes, num_nodes - (num_nodes + num_edge_types)

        unique_start = self.num_nodes + self.num_edge_types
        self.CLS, self.SEP1, self.SEP2, self.MASK, self.END = torch.arange(unique_start, unique_start + 5, 1)
        self.ntokens = self.END + 1

        self.trX, self.trY = self.get_masked_data(train_data)

        valX, valY = self.get_masked_data(val_data)
        valX2, unique_triplet_indices = torch.unique(valX, dim=0, return_inverse=True)
        total_unique_indices = torch.unique(unique_triplet_indices)
        unique_sets = []
        for i in total_unique_indices:
            one_hot = F.one_hot(valY[unique_triplet_indices == i], num_classes=self.num_nodes)
            unique_sets.append(one_hot.sum(dim=0)[None,:])
        valY2 = torch.cat(unique_sets)
        self.valX, self.valY = valX2, valY2

        teX, teY = self.get_masked_data(test_data)
        teX2, unique_triplet_indices = torch.unique(teX, dim=0, return_inverse=True)
        total_unique_indices = torch.unique(unique_triplet_indices)
        unique_sets = []

        for i in total_unique_indices:
            one_hot = F.one_hot(teY[unique_triplet_indices == i], num_classes=self.num_nodes)
            unique_sets.append(one_hot.sum(dim=0)[None,:])    
        teY2 = torch.cat(unique_sets)
        self.teX, self.teY = teX2, teY2

    def get_masked_data(self, data):
        num_edges = data.edge_index.shape[1]

        data_masked = [
            torch.zeros((num_edges, 1), device=device).fill_(self.CLS),
            data.edge_index[0][:, None],   # s
            torch.zeros((num_edges, 1), device=device).fill_(self.SEP1),
            data.edge_type[:, None] + self.num_nodes,  # r
            torch.zeros((num_edges, 1), device=device).fill_(self.SEP2),
            torch.zeros((num_edges, 1), device=device).fill_(self.MASK), # o, masked
            torch.zeros((num_edges, 1), device=device).fill_(self.END)
        ]
        data_masked = torch.hstack(data_masked)
        labels_masked = data.edge_index[1][:, None]

        return data_masked, labels_masked
    
    def get_data(self):
        return self.trX, self.trY, self.valX, self.valY, self.teX, self.teY


class TransformerModel(nn.Module):
    def __init__(self, ntokens, d_model, nhead, d_out, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(ntokens, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, d_out)

    def forward(self, src):
        src_embed = self.embedding(src)
        output = self.transformer(src_embed)
        output = self.fc(output)
        return output


def batchify(data, bsz, samp_override=None):
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N, dim]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz, dim]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(seq_len, bsz, -1)
    return data.to(device)[:samp_override]


def train(model, verbose=False):
    model.train()

    losses = []
    for i in tqdm(range(len(trX)), disable=not verbose):
        X, Y = trX[i].long(), trY[i].long().squeeze()

        output = model(X)[:, -2]
        # print(output, Y.shape); raise
        loss = criterion(output.view(-1, main_dataset.num_nodes), Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return np.mean(losses)

# train(model)


@torch.no_grad()
def eval(model, dataX, dataY):
    model.eval()
    
    hits_1, hits_10, mRR, mAP = [], [], [], []

    for i in range(len(dataX)):
        X, Y = dataX[i].long(), dataY[i].long().squeeze()
        output = model(X)[:, -2].view(-1, main_dataset.num_nodes)

        vals, indices = torch.topk(output, 10, dim=1)
        indices_1 = indices[:, :1]
        hits_10.append(torch.gather(Y, 1, indices).sum(dim=1).float().mean().item())
        hits_1.append(torch.gather(Y, 1, indices_1).sum(dim=1).float().mean().item())

        # mRR computation
        ranked_output_idx = torch.argsort(output, descending=True)
        ranked_output = torch.gather(Y, 1, ranked_output_idx)
        mRR.append((1 / (ranked_output.argmax(dim=1)+1)).mean().item())

        # mAP computation
        cumulative_sum = (torch.cumsum(ranked_output, dim=1) * ranked_output)
        div_factor = torch.arange(main_dataset.num_nodes).to(device)
        cumulative_sum = cumulative_sum.float() / (div_factor + 1)
        cumulative_sum = cumulative_sum / torch.sum(ranked_output, dim=1).unsqueeze(1)
        mAP.append(cumulative_sum.sum(dim=1).mean().item())

    hits_1 = np.mean(hits_1)
    hits_10 = np.mean(hits_10)
    mRR = np.mean(mRR)
    mAP = np.mean(mAP)

    return hits_1, hits_10, mRR, mAP

main_dataset = KGDataset(dataset)

batch_size_train = 200
batch_size_test = 50

batch_sizes = [batch_size_train] * 2 + [batch_size_test] * 4

trX, trY, valX, valY, teX, teY = [batchify(d, bsz, samp_override=None) for d, bsz in zip(main_dataset.get_data(), batch_sizes)]


ntokens = main_dataset.ntokens
model = TransformerModel(ntokens, d_model=256, nhead=8, d_out=main_dataset.num_nodes, num_layers=8).to(device)

criterion = nn.CrossEntropyLoss()
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)


epochs = args.n_epochs

for i in tqdm(range(epochs)):
    loss = train(model, verbose=False)
    if i==0: tqdm.write("train loss, hits@1, hits@10, mRR, mAP")
    tqdm.write(f"{loss:.4f} " + " " + " ".join([str(i.__format__(".6f")) for i in eval(model, valX, valY)]))

hits_1, hits_10, mRR, mAP = eval(model, teX, teY)
print(f"Hits@1: {hits_1:.4f}, Hits@10: {hits_10:.4f}, MRR: {mRR:.4f}, MAP: {mAP:.4f}")