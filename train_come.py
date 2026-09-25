import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from come import TemporalStability, adaptation_loss, allocate_experts, memory_pseudo_labels, perturbed_views, structural_stability
from datasets import CSBMDataset, CitationDataset, TwitchDataset
from model import NodeClassificationModel, TargetAdaptiveModel
SEEDS = (2080, 2090, 2100, 2110, 2120)
SOURCES = {'CSBM-G4': ('CSBM', ('CSBM-G1', 'CSBM-G2', 'CSBM-G3')), 'DE': ('Twitch', ('RU', 'PTBR', 'FR', 'ES')), 'EN': ('Twitch', ('RU', 'PTBR', 'FR', 'ES')), 'DBLPv7': ('Citation', ('ACMv9', 'Citationv1')), 'ACMv9': ('Citation', ('DBLPv7', 'Citationv1')), 'Citationv1': ('Citation', ('DBLPv7', 'ACMv9'))}
DATASETS = {'CSBM': CSBMDataset, 'Twitch': TwitchDataset, 'Citation': CitationDataset}

def run(target, config, seed, device, data_root, checkpoint_root):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    family, source_names = SOURCES[target]
    dataset = DATASETS[family](str(data_root / family / target), target)
    data = dataset[0].to(device)
    first_checkpoint = torch.load(checkpoint_root / f'model_{source_names[0]}.pth', map_location='cpu', weights_only=True)
    settings = SimpleNamespace(gnn='gcn', nhid=128, num_layers=2, use_bn=False, dropout_ratio=config['dropout'], num_features=data.x.size(1), num_classes=first_checkpoint['gnn.cls.bias'].numel(), src=list(source_names), lambda_tradeoff=config['lambda_g'], cache_static_first_layer=False, uniform_attention_init=False)
    sources = []
    for name in source_names:
        source = NodeClassificationModel(settings).to(device)
        source.load_state_dict(torch.load(checkpoint_root / f'model_{name}.pth', map_location=device, weights_only=True))
        source.eval().requires_grad_(False)
        sources.append(source)
    weights = [[dict(source.gnn.named_parameters())[f'convs.{layer}.lin.weight'] for source in sources] for layer in range(settings.num_layers)]
    model = TargetAdaptiveModel(settings, weights, sources).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    temporal = TemporalStability(config['rho'])
    model.eval()
    with torch.no_grad():
        memory_features = model.feat_bottleneck(data.x, data.edge_index).detach()
        memory_predictions = torch.stack([source(data.x, data.edge_index).exp() for source in sources]).mean(dim=0)
    reliability = source_mean = source_prediction = None
    for epoch in range(config['epochs']):
        if epoch % config['probe_interval'] == 0:
            views = perturbed_views(data, count=config['K'], drop=config['edge_drop'], add=config['edge_add'])
            reliability, source_mean, source_prediction = structural_stability(sources, data, views)
        model.train()
        features = model.feat_bottleneck(data.x, data.edge_index)
        adaptive = F.softmax(model.feat_classifier(features), dim=1)
        with torch.no_grad():
            pseudo_label, pseudo_distribution = memory_pseudo_labels(features.detach(), memory_features, memory_predictions, config['K_nn'])
            stable, conflicting, ambiguous = temporal.update(adaptive.detach(), source_mean, pseudo_label)
        loss = adaptation_loss(adaptive, pseudo_label, pseudo_distribution, source_prediction, stable, conflicting, config['lambda_sta'], config['lambda_con'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            memory_features = config['memory_gamma'] * memory_features + (1 - config['memory_gamma']) * features.detach()
            memory_predictions = config['memory_gamma'] * memory_predictions + (1 - config['memory_gamma']) * adaptive.detach()
        if (epoch + 1) % 100 == 0:
            print(f'{target} seed={seed} epoch={epoch + 1} loss={loss.item():.4f}', flush=True)
    model.eval()
    with torch.no_grad():
        adaptive = model(data.x, data.edge_index).exp()
        pseudo_label, _ = memory_pseudo_labels(model.feat_bottleneck(data.x, data.edge_index), memory_features, memory_predictions, config['K_nn'])
        stable, conflicting, ambiguous = temporal.update(adaptive, source_mean, pseudo_label)
        final, _ = allocate_experts(reliability, source_mean, adaptive, stable, conflicting, ambiguous, temporal.agreement)
        accuracy = (final.argmax(dim=1) == data.y).float().mean().item() * 100
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=SOURCES, required=True)
    parser.add_argument('--config', default='configs/paper.json')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--checkpoint-root', type=Path, default=Path('pretrain'))
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    config = settings['common'] | settings['targets'][args.target]
    seeds = (args.seed,) if args.seed is not None else SEEDS
    scores = [run(args.target, config, seed, torch.device(args.device), args.data_root, args.checkpoint_root) for seed in seeds]
    print(f'{args.target}: {np.mean(scores):.2f} ± {np.std(scores):.2f} ({len(scores)} runs)')
if __name__ == '__main__':
    main()
