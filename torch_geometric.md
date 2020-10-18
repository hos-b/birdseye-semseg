# Torch Geometric Library
a contribution to torch, not part of the main distribution (yet).


## Typical Model
the model still inherits from `torch.nn.module`. it seems like the entire stack of a GNN is defined in a single class.
as with normal DNN models, the layers are defined in the `__init__` function, however the nature of the layers is different.

## Graph Convolution Layers
`geonn.GCNConv` and `geonn.GINConv` are instances of `MessagePassing` (which is a subclass of nn.Module). they define a single layer of graph convolution, which can be decomposed into
* message computation
* aggregation
* update
* pooling

one can define custom graph convolution layers by inheriting from MessagePassing and using its key blocks:
* `aggr='add'`: the aggeration strategy: "add", "mean" or "max
* `propagate()`: the initial call to start propagating messages. takes in the edge indices and any other data to pass along (e.g. to update node embeddings)
* `message()`: constructs messages to node i. takes in any argument which was originally passed to `propagate()`
* `update()`: updates node embeddings. takes in the output of aggregation as first argument, and any argument which was originally passed to `propagate()`

### Custom Convolution Class
```python
class CustomConv(geonn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x is [N, in_channels]
        # edge index is [2, E]
        # add self loops to adjacency list
        edge_index, _ = geoutils.add_self_loops(edge_index, num_nodes=x.size(0))
        # transform feature matrix
        x = self.lin(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # compute messages
        # x_j is [E, out_channels]
        row, col = edge_index
        # GCN Conv
        deg = geoutils.degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j
```

## Forward Function
the forward function receives a batch of graph data that is composed of:
* `data.x`: an `Nxd` matrix that contains the node features.
* `data.edge_index` :an adjacency list(not matrix) that contains the edges between the given nodes
* `data.batch`: an `Nx1` list that shows which graph in the dataset, each given node belongs to. for typical node classification, this is trivial because the task is usually performed on one graph (at a time).

### sanity check
a sanity check can be done to see if `data.num_node_features` is zero. if so, an `Nx1` matrix of ones replaces `data.x`.

### graph convolution
inside the forward function, there is a for-loop that passes the nodes and the adjacency list through the predefined graph convolution layers.
in the below snippet, self.convs is a module list, made of graph convolution layers
```python
for i in range(self.num_layers):
    x = self.convs[i](x, edge_index)
    x = F.dropout(x, p=self.dropout_rate, training=self.training)
```
`self.num_layers` defines the depth or the number of hops we go over. `x` is fed into the next hop inside the loop.
### graph pooling
if the task is graph-wide, e.g. graph classification, a gloabl mean pooling or max pooling can be done.
```python
x = geonn.global_mean_pool(x, batch)
```

### linear layers
after the graph convolution layers, we can have a bunch of linear layers.

## Training

### train mask
if the task is node classification, we only have one graph and we don't want to look at parts of it that could be in the test set.
that's why torch geometric allows us to use the calculated training mask:
```python
pred = model(batch)
if task == 'node':
    pred = pred[batch.train_mask]
    label = batch.y[batch.train_mask]
loss = F.nll_loss(pred, label)
loss.backward()
```