from typing import Union, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing, SimpleConv
from torch_geometric.utils import add_remaining_self_loops

class Sparsemax(nn.Module):

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        z = x - x.max(dim=self.dim, keepdim=True).values
        sorted_z = z.sort(dim=self.dim, descending=True).values
        ranks = torch.arange(1, z.size(self.dim) + 1, device=z.device, dtype=z.dtype)
        shape = [1] * z.ndim
        shape[self.dim] = -1
        ranks = ranks.view(shape)
        cumulative = sorted_z.cumsum(dim=self.dim) - 1
        support = ranks * sorted_z > cumulative
        k = support.sum(dim=self.dim, keepdim=True)
        threshold = cumulative.gather(self.dim, k - 1) / k
        return (z - threshold).clamp_min(0)

class NodeCentricConv(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, model_weights: tuple=(), lambda_tradeoff: float=0.2, cache_source: bool=False, uniform_attention_init: bool=False, aggr: str='mean', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lambda_tradeoff = lambda_tradeoff
        self.cache_source = cache_source
        self._source_cache_key = None
        self._source_cache = None
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.films = list()
        for weight in model_weights:
            self.films.append(weight.t())
        self.att = Parameter(torch.Tensor(self.out_channels, 1))
        if uniform_attention_init:
            nn.init.zeros_(self.att)
        else:
            nn.init.xavier_normal_(self.att)
        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        nn.init.xavier_normal_(self.weight)
        self.neigh_aggr = SimpleConv(aggr='mean')
        self.sparse_attention = Sparsemax(dim=1)

    def _cached_source_terms(self, x: Tensor, edge_index: Adj):
        key = (x.data_ptr(), edge_index.data_ptr(), tuple(x.size()), tuple(edge_index.size()), str(x.device))
        if self._source_cache_key == key and self._source_cache is not None:
            return self._source_cache
        edge_index_loop, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        with torch.no_grad():
            neigh_rep = self.neigh_aggr(x, edge_index_loop).detach()
            base_out = self.propagate(edge_index_loop, x=x, gamma=torch.sigmoid(neigh_rep), edge_weight=None, size=None).detach()
            att_inputs = [torch.matmul(neigh_rep, film).detach() for film in self.films]
            source_reps = [torch.matmul(base_out, film).detach() for film in self.films]
        self._source_cache_key = key
        self._source_cache = (neigh_rep, att_inputs, source_reps)
        return self._source_cache

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_type: OptTensor=None) -> Tensor:
        use_cache = self.cache_source and all((not film.requires_grad for film in self.films))
        if use_cache:
            neigh_rep, att_inputs, reps = self._cached_source_terms(x, edge_index)
            atts = [torch.matmul(rep, self.att) for rep in att_inputs]
        else:
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
            neigh_rep = self.neigh_aggr(x, edge_index)
            atts = []
            reps = []
            out = self.propagate(edge_index, x=x, gamma=torch.sigmoid(neigh_rep), edge_weight=None, size=None)
            for i, film in enumerate(self.films):
                rep = torch.matmul(neigh_rep, film)
                res = torch.matmul(rep, self.att)
                atts.append(res)
                rep = torch.matmul(out, film)
                reps.append(rep)
        atts = torch.cat(atts, dim=1)
        w = self.sparse_attention(atts)
        gamma = torch.stack(reps)
        w = w.t().unsqueeze(-1)
        gamma = torch.sum(w * gamma, dim=0)
        if self.lambda_tradeoff == 0:
            out = gamma
        else:
            wg = torch.matmul(neigh_rep, self.weight)
            out = gamma + wg * self.lambda_tradeoff
        return out

    def message(self, x_j: Tensor, gamma_i: Tensor, edge_weight: OptTensor) -> Tensor:
        out = gamma_i * x_j
        return out

class MLPModule(torch.nn.Module):

    def __init__(self, args, model_list):
        super(MLPModule, self).__init__()
        self.args = args
        self.model_list = model_list
        self.att = Parameter(torch.Tensor(args.num_classes, 1))
        if getattr(args, 'uniform_attention_init', False):
            nn.init.zeros_(self.att)
        else:
            nn.init.xavier_normal_(self.att)
        self.sparse_attention = Sparsemax(dim=1)

    def forward(self, x):
        outputs = []
        weights = []
        for i in range(len(self.model_list)):
            cls_output = self.model_list[i].gnn.cls(x)
            att = torch.matmul(cls_output, self.att)
            outputs.append(cls_output)
            weights.append(att)
        weights = torch.cat(weights, dim=1)
        w = self.sparse_attention(weights)
        outputs = torch.stack(outputs)
        w = w.t().unsqueeze(-1)
        x = torch.sum(w * outputs, dim=0)
        return x
