import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from layer import NodeCentricConv, MLPModule

class GCN(nn.Module):

    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(self.num_features, self.nhid))
        self.bns.append(nn.BatchNorm1d(self.nhid))
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.nhid, self.nhid))
            self.bns.append(nn.BatchNorm1d(self.nhid))
        self.cls = torch.nn.Linear(self.nhid, self.num_classes)
        self.activation = F.relu
        self.use_bn = args.use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.feat_bottleneck(x, edge_index, edge_weight)
        x = self.feat_classifier(x)
        return x

    def feat_bottleneck(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        return x

    def feat_classifier(self, x):
        x = self.cls(x)
        return x

class NodeClassificationModel(torch.nn.Module):

    def __init__(self, args):
        super(NodeClassificationModel, self).__init__()
        self.args = args
        self.gnn = GCN(args)
        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index):
        x = self.feat_bottleneck(x, edge_index)
        x = self.feat_classifier(x)
        return F.log_softmax(x, dim=1)

    def feat_bottleneck(self, x, edge_index):
        x = self.gnn.feat_bottleneck(x, edge_index)
        return x

    def feat_classifier(self, x):
        x = self.gnn.feat_classifier(x)
        return x

class TargetAdaptiveModel(torch.nn.Module):

    def __init__(self, args, model_weights, model_list):
        super(TargetAdaptiveModel, self).__init__()
        self.args = args
        self.src = args.src
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.model_weights = model_weights
        self.model_list = model_list
        self.lambda_tradeoff = getattr(args, 'lambda_tradeoff', 0.2)
        self.cache_static_first_layer = getattr(args, 'cache_static_first_layer', False)
        self.uniform_attention_init = getattr(args, 'uniform_attention_init', False)
        self.num_layers = args.num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(NodeCentricConv(self.num_features, self.nhid, model_weights[0], lambda_tradeoff=self.lambda_tradeoff, cache_source=self.cache_static_first_layer, uniform_attention_init=self.uniform_attention_init))
        self.bns.append(nn.BatchNorm1d(self.nhid))
        for i in range(self.num_layers - 1):
            self.convs.append(NodeCentricConv(self.nhid, self.nhid, model_weights[i + 1], lambda_tradeoff=self.lambda_tradeoff, uniform_attention_init=self.uniform_attention_init))
            self.bns.append(nn.BatchNorm1d(self.nhid))
        self.activation = F.relu
        self.use_bn = args.use_bn
        self.cls = MLPModule(args, model_list)

    def forward(self, x, edge_index):
        x = self.feat_bottleneck(x, edge_index)
        x = self.feat_classifier(x)
        return F.log_softmax(x, dim=1)

    def feat_bottleneck(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        return x

    def feat_classifier(self, x):
        x = self.cls(x)
        return x
