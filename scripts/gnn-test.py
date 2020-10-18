#pylint: disable=E1101
#pylint: disable=not-callable
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch geometric
import torch_geometric.nn as geonn
import torch_geometric.utils as geoutils
import torch_geometric.transforms as GeoT

# visaulization of the graph
import networkx as nx
import numpy as np

# datasets
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader as GeoDataLoader

# other stuff
import time
from datetime import datetime
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.gemoetric import GNNStack

batch_size = 32
learning_rate = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = TUDataset(root='./data', name='IMDB-BINARY')
task = 'graph'
# dataset = Planetoid(root='./data', name='citeseer')
# task = 'node'
dataset_size = len(dataset)
loader = GeoDataLoader(dataset[:int(dataset_size * 0.8)], batch_size=batch_size, shuffle=True)
test_loader = GeoDataLoader(dataset[int(dataset_size * 0.8):], batch_size=batch_size, shuffle=True)

model = GNNStack(max(dataset.node_features, 1), 32, dataset.num_classes, task)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(200):
    total_loss = 0
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        embedding, pred = model(batch)
        label = batch.y
        if task == 'node':
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
        loss = F.nll_loss(pred, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    total_loss /= dataset_size
    print(f"loss = {total_loss}")