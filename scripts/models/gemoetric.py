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

class GNNStack(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
		super(GNNStack, self).__init__()
		if not (task=='node' or task=='graph'):
			raise RuntimeError("unknown graph task")
		self.task = task
		self.convs = nn.ModuleList()
		self.convs.append(self.build_conv_model(input_dim, hidden_dim))
		self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
		self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

		# post message passing
		self.post_mp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
									 nn.Linear(hidden_dim, output_dim))
		self.dropout_rate = 0.25
		self.num_layers = 3

	def build_conv_model(self, input_dim, hidden_dim):
		if self.task == 'node':
			return geonn.GCNConv(input_dim, hidden_dim)
		else:
			return geonn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
											   nn.ReLU(),
											   nn.Linear(hidden_dim, hidden_dim)))

	def forward(self, data):
		embedding = None
		# data.x: nodes x features
		# data.edge_index: sparse adjacency list, e.g. [(1,2), (1, 4), ...]
		# data.batch: nodes x 1, which graph the node belongs to, e.g. [1, 1, 1, 1, 2, 2, 2, ...]
		# 			  for node classification, it all runs on one graph, so it's trivial
		x, edge_index, batch = data.x, data.edge_index, data.batch
		if data.num_node_features == 0:
			x = torch.ones(data.num_nodes, 1)
		
		for i in range(self.num_layers):
			x = self.convs[i](x, edge_index)
			embedding = x
			x = F.dropout(x, p=self.dropout_rate, training=self.training)

		if self.task == 'graph':
			x = geonn.global_mean_pool(x, batch)
		
		x = self.post_mp(x)
		return embedding, F.log_softmax(x, dim=1)