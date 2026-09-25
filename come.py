import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge, negative_sampling, to_undirected
from layer import Sparsemax
EPS = 1e-12

def perturbed_views(data, count=5, drop=0.1, add=0.05):
    edges = data.edge_index
    views = [edges]
    for index in range(count):
        if index % 2 == 0:
            view, _ = dropout_edge(edges, p=drop, force_undirected=True, training=True)
        else:
            n = max(1, int(edges.size(1) * add))
            extra = negative_sampling(edges, num_nodes=data.num_nodes, num_neg_samples=n)
            view = to_undirected(torch.cat([edges, extra], dim=1), num_nodes=data.num_nodes)
        views.append(view)
    return views

@torch.no_grad()
def structural_stability(source_models, data, views):
    predictions = torch.stack([torch.stack([model(data.x, edge).exp() for edge in views], dim=1) for model in source_models], dim=1)
    confidence = predictions.max(dim=-1).values
    mean_confidence = confidence.mean(dim=2)
    variation = confidence.std(dim=2, unbiased=False)
    labels = predictions.argmax(dim=-1)
    consistency = (labels == labels.mode(dim=2).values.unsqueeze(-1)).float().mean(dim=2)
    confidence_drop = (confidence[:, :, 0] - mean_confidence).clamp_min(0)
    row, col = data.edge_index
    neighborhood = torch.zeros_like(mean_confidence)
    degree = torch.zeros(data.num_nodes, device=predictions.device)
    neighborhood.index_add_(0, col, confidence[:, :, 0][row])
    degree.index_add_(0, col, torch.ones_like(col, dtype=degree.dtype))
    neighborhood = neighborhood / degree.clamp_min(1).unsqueeze(1)
    trend = (neighborhood - confidence[:, :, 0]).clamp_min(0)
    trend = torch.where(degree[:, None] > 0, trend, 0)
    reliability = torch.exp(-(1 - mean_confidence + variation + (1 - consistency) + confidence_drop + trend))
    source_mean = predictions.mean(dim=2)
    source_weights = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(EPS)
    source_prediction = (source_weights.unsqueeze(-1) * source_mean).sum(dim=1)
    return (reliability, source_mean, source_prediction)

@torch.no_grad()
def memory_pseudo_labels(features, memory_features, memory_predictions, neighbors):
    similarity = F.normalize(features, dim=1) @ F.normalize(memory_features, dim=1).T
    similarity.fill_diagonal_(-torch.inf)
    indices = similarity.topk(min(neighbors, similarity.size(1) - 1), dim=1).indices
    distribution = memory_predictions[indices].mean(dim=1)
    return (distribution.argmax(dim=1), distribution)

class TemporalStability:

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.conflict = None
        self.agreement = None

    @torch.no_grad()
    def update(self, adaptive, source_mean, pseudo_label):
        disagreement = 1 - adaptive.gather(1, pseudo_label[:, None]).squeeze(1)
        agreement = source_mean.gather(2, pseudo_label[:, None, None].expand(-1, source_mean.size(1), 1)).squeeze(2)
        if self.conflict is None:
            self.conflict, self.agreement = (disagreement, agreement)
        else:
            self.conflict = self.momentum * self.conflict + (1 - self.momentum) * disagreement
            self.agreement = self.momentum * self.agreement + (1 - self.momentum) * agreement
        stable = 1 - self.conflict
        conflicting = self.conflict
        ambiguous = 4 * self.conflict * (1 - self.conflict)
        return (stable, conflicting, ambiguous)

def adaptation_loss(adaptive, pseudo_label, pseudo_distribution, source_prediction, stable, conflicting, lambda_sta=1.0, lambda_con=0.05):
    logp = adaptive.clamp_min(EPS).log()
    entropy = -(adaptive * logp).sum(dim=1).mean()
    marginal = adaptive.mean(dim=0)
    information = entropy + (marginal * marginal.clamp_min(EPS).log()).sum()
    target = 0.5 * (pseudo_distribution + source_prediction)
    stable_loss = (stable * F.nll_loss(logp, pseudo_label, reduction='none')).mean()
    conflict_loss = (conflicting * target.max(dim=1).values * -(target * logp).sum(dim=1)).mean()
    return information + lambda_sta * stable_loss + lambda_con * conflict_loss

@torch.no_grad()
def allocate_experts(reliability, source_mean, adaptive, stable, conflicting, ambiguous, agreement):
    uncertainty = -(source_mean * source_mean.clamp_min(EPS).log()).sum(dim=2)
    uncertainty_r = -reliability.clamp_min(EPS).log()
    normalized = (uncertainty_r - uncertainty_r.min(dim=1, keepdim=True).values) / (uncertainty_r.max(dim=1, keepdim=True).values - uncertainty_r.min(dim=1, keepdim=True).values).clamp_min(EPS)
    u = torch.maximum(conflicting, ambiguous).unsqueeze(1)
    state_reliability = reliability * (agreement + (1 - agreement) * u)
    support = ambiguous.unsqueeze(1) * (1 + normalized)
    source_scores = (state_reliability + EPS).log() + support - uncertainty
    adaptive_score = (1 - state_reliability.max(dim=1).values + EPS).log() - support.mean(dim=1) + uncertainty.mean(dim=1)
    allocation = Sparsemax(dim=1)(torch.cat([source_scores, adaptive_score[:, None]], dim=1))
    final = (allocation[:, :-1, None] * source_mean).sum(dim=1) + allocation[:, -1, None] * adaptive
    return (final, allocation)
