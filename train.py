import argparse
import copy
import csv
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.RTSRNet import RTSRNet, VALID_ABLATION_MODES, ABLATION_DIR_NAMES
from utils.config import Config
from utils.edge_dataloader import TrainEdgeDataset
from utils.utils import CosineDecay


LOSS_POLICY = {
    'mask': 'mean(structure_loss(P2..P5, GT))',
    'edge': 'BCEWithLogits(Pe, edge_GT)',
    'paper_total': 'mask + 0.05 * edge',
    'legacy_control': 'optional; enabled only with --legacy_control_loss',
}


def parse_args():
    p = argparse.ArgumentParser(description='Train RTSRNet for depth-assisted camouflaged object detection.')
    for k in ['epochs', 'batch_size', 'trainsize', 'num_workers']:
        p.add_argument(f'--{k}', type=int, default=None)
    p.add_argument('--dataset_dir', type=str, default=None)
    p.add_argument('--save_dir', type=str, default='')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', action='store_true', default=False)
    p.add_argument('--no_ema', action='store_true', default=False)
    p.add_argument('--ablation_mode', type=str, default='full', choices=VALID_ABLATION_MODES,
                   help='Ablation switch: baseline, dgp, dgp_dls, idls, goodr, or full.')
    p.add_argument('--rel_loss_weight', type=float, default=0.05,
                   help='Weight of the optional legacy TGMM control loss (only used with --legacy_control_loss).')
    p.add_argument('--rel_depth_thr', type=float, default=0.25,
                   help='Depth-edge threshold for false-depth-edge negatives.')
    p.add_argument('--rel_pos_weight', type=float, default=0.5,
                   help='Positive GT-boundary reliability weight.')
    p.add_argument('--rel_neg_weight', type=float, default=2.0,
                   help='Negative false-depth-edge reliability weight.')
    p.add_argument('--no_rel_loss', action='store_true', default=False,
                   help='Disable the legacy TGMM control supervision while keeping the legacy routing.')
    p.add_argument('--legacy_control_loss', action='store_true', default=False,
                   help='Re-enable the auxiliary control loss used by the pre-refactor GoodRT_v2 training code. '
                        'The manuscript-facing default uses only mask loss + 0.05 * edge loss.')
    return p.parse_args()

def make_save_dir(cfg, args):
    if args.save_dir:
        return args.save_dir
    return os.path.join(cfg.result_path, ABLATION_DIR_NAMES[args.ablation_mode])

def structure_loss(logits, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, 1, 15) - mask)
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
    wbce = (weit * wbce).sum((2, 3)) / weit.sum((2, 3))
    pred = torch.sigmoid(logits)
    inter = ((pred * mask) * weit).sum((2, 3))
    union = ((pred + mask) * weit).sum((2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def edge_gt_loss(edge_logits, edge_gt):
    return F.binary_cross_entropy_with_logits(edge_logits, edge_gt)


def depth_geometry_like(depth, size=None, eps=1e-6):
    sobel_x = torch.tensor(
        [[[[-1.0, 0.0, 1.0],
           [-2.0, 0.0, 2.0],
           [-1.0, 0.0, 1.0]]]],
        device=depth.device,
        dtype=depth.dtype,
    )
    sobel_y = torch.tensor(
        [[[[-1.0, -2.0, -1.0],
           [ 0.0,  0.0,  0.0],
           [ 1.0,  2.0,  1.0]]]],
        device=depth.device,
        dtype=depth.dtype,
    )
    gx = F.conv2d(depth, sobel_x, padding=1)
    gy = F.conv2d(depth, sobel_y, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + eps)
    grad = grad / (grad.amax(dim=(2, 3), keepdim=True) + eps)
    if size is not None and grad.shape[-2:] != size:
        grad = F.interpolate(grad, size=size, mode='bilinear', align_corners=False)
    return grad.clamp(0.0, 1.0)


def reliability_loss_for_logits(logits, gt, depth, edge_gt,
                                depth_thr=0.25, pos_weight=0.5, neg_weight=2.0):
    """Reliability supervision without extra annotations.

    Positive: GT boundary nearby, light weight.
    Negative: strong depth edge far from GT boundary, strong weight.
    This matches suppress-only GoodR: the main goal is to darken false depth edges.
    """
    if logits is None:
        return None

    size = logits.shape[-2:]
    edge = F.interpolate(edge_gt, size=size, mode='bilinear', align_corners=False).clamp(0.0, 1.0)
    dep_edge = depth_geometry_like(depth, size=size)

    pos = (edge > 0.10).float()
    neg = ((edge < 0.05) & (dep_edge > depth_thr)).float()
    valid = ((pos + neg) > 0).float()
    if float(valid.sum().detach().cpu()) < 1.0:
        return logits.new_tensor(0.0)

    target = pos
    weight = pos * pos_weight + neg * neg_weight
    loss_map = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (loss_map * weight * valid).sum() / ((weight * valid).sum() + 1e-6)


def tgmm_reliability_loss(model, gt, depth, edge_gt,
                          depth_thr=0.25, pos_weight=0.5, neg_weight=2.0):
    """DLS-only reliability loss: only tgmm4/tgmm3/tgmm2 are supervised."""
    modules = []
    if hasattr(model, 'decoder'):
        modules.extend([
            getattr(model.decoder, 'tgmm4', None),
            getattr(model.decoder, 'tgmm3', None),
            getattr(model.decoder, 'tgmm2', None),
        ])

    losses = []
    for mod in modules:
        logits = getattr(mod, 'last_reliability_logits', None) if mod is not None else None
        loss = reliability_loss_for_logits(
            logits, gt, depth, edge_gt,
            depth_thr=depth_thr,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
        )
        if loss is not None:
            losses.append(loss)
    if not losses:
        return gt.new_tensor(0.0)
    return torch.stack(losses).mean()


def target_prior_loss_for_logits(logits, gt):
    """Supervise GoodRT target prior without adding a new manual loss weight."""
    if logits is None:
        return None
    size = logits.shape[-2:]
    target = F.interpolate(gt, size=size, mode='bilinear', align_corners=False).clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(logits, target)


def tgmm_target_prior_loss(model, gt):
    modules = []
    if hasattr(model, 'decoder'):
        modules.extend([
            getattr(model.decoder, 'tgmm4', None),
            getattr(model.decoder, 'tgmm3', None),
            getattr(model.decoder, 'tgmm2', None),
        ])
    losses = []
    for mod in modules:
        logits = getattr(mod, 'last_target_prior_logits', None) if mod is not None else None
        loss = target_prior_loss_for_logits(logits, gt)
        if loss is not None:
            losses.append(loss)
    if not losses:
        return gt.new_tensor(0.0)
    return torch.stack(losses).mean()


def tgmm_control_loss(model, gt, depth, edge_gt,
                      depth_thr=0.25, pos_weight=0.5, neg_weight=2.0):
    """GoodRT liquid-control auxiliary loss.

    Reuses the existing rel_loss_weight. No new manual loss weight is added.
    """
    rel = tgmm_reliability_loss(
        model, gt, depth, edge_gt,
        depth_thr=depth_thr,
        pos_weight=pos_weight,
        neg_weight=neg_weight,
    )
    tgt = tgmm_target_prior_loss(model, gt)
    return 0.5 * (rel + tgt)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(save_dir)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(save_dir, 'train.log'), mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    return logger


def atomic_save(obj, save_path):
    tmp = save_path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, save_path)


def main():
    args = parse_args()
    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.trainsize is not None:
        cfg.trainsize = args.trainsize
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.dataset_dir:
        cfg.dp.dataset_dir = args.dataset_dir
        cfg.dp.refresh()

    save_dir = make_save_dir(cfg, args)
    logger = build_logger(save_dir)
    set_seed(args.seed)
    device = cfg.device

    logger.info('Model: RTSRNet')
    logger.info('Ablation mode: %s', args.ablation_mode)
    logger.info('Seed: %d', args.seed)
    logger.info('Loss policy: %s', json.dumps(LOSS_POLICY, ensure_ascii=False))

    model = RTSRNet(pretrained=True, ablation_mode=args.ablation_mode).to(device)
    bp, op = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bp if 'encoder' in n else op).append(p)

    optimizer = torch.optim.AdamW(
        [
            {'params': bp, 'lr': cfg.learning_rate * cfg.backbone_lr_mult},
            {'params': op, 'lr': cfg.learning_rate},
        ],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineDecay(optimizer=optimizer, max_lr=cfg.learning_rate, min_lr=cfg.min_lr, max_epoch=cfg.epochs)
    loader = DataLoader(
        TrainEdgeDataset(
            cfg.dp.train_imgs,
            cfg.dp.train_masks,
            cfg.dp.train_depth,
            cfg.dp.train_edges,
            cfg.trainsize,
            True,
            True,
            True,
            True,
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and torch.cuda.is_available())
    ema = None if args.no_ema else ModelEMA(model, cfg.ema_decay)
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'ema_model' in ckpt and ema is not None:
            ema.load_state_dict(ckpt['ema_model'])
        if 'scheduler_epoch' in ckpt:
            for _ in range(int(ckpt['scheduler_epoch'])):
                scheduler.step()
        start_epoch = int(ckpt.get('epoch', 0)) + 1

    config_snapshot = vars(args).copy()
    config_snapshot.update({
        'model': 'RTSRNet',
        'ablation_mode': args.ablation_mode,
        'active_modules': {
            'legacy_geometry_calibration': args.ablation_mode in ('dgp', 'dgp_dls', 'idls', 'goodr', 'full'),
            'legacy_decoder_state_update': args.ablation_mode in ('dgp_dls', 'idls', 'goodr', 'full'),
            'legacy_iterative_update': args.ablation_mode in ('idls', 'goodr', 'full'),
            'legacy_reliability_routing': args.ablation_mode in ('goodr', 'full'),
            'legacy_state_readout': args.ablation_mode == 'full',
        },
        'dataset_dir': cfg.dp.dataset_dir,
        'epochs': cfg.epochs,
        'batch_size': cfg.batch_size,
        'trainsize': cfg.trainsize,
        'loss_policy': LOSS_POLICY,
        'checkpoint_selection': 'minimum_training_loss',
        'rel_loss_weight': args.rel_loss_weight,
        'rel_depth_thr': args.rel_depth_thr,
        'rel_pos_weight': args.rel_pos_weight,
        'rel_neg_weight': args.rel_neg_weight,
        'legacy_control_loss_enabled': args.legacy_control_loss and args.ablation_mode in ('goodr', 'full') and (not args.no_rel_loss) and args.rel_loss_weight > 0,
        'checkpoint_compatibility': 'paper-facing names with legacy GoodRT_v2 state_dict remapping',
    })
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    best_loss = float('inf')
    metrics_csv = os.path.join(save_dir, 'train_epoch_metrics.csv')
    if start_epoch == 1 and not os.path.exists(metrics_csv):
        with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'total', 'seg', 'edge_gt', 'legacy_tgmm_control', 'lr'])

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        running = {'total': 0.0, 'seg': 0.0, 'egt': 0.0, 'rel': 0.0}
        pbar = tqdm(loader, desc=f'Epoch {epoch}/{cfg.epochs}')
        for _, img, gt, dep, edge_gt in pbar:
            img = img.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            dep = dep.to(device, non_blocking=True)
            edge_gt = edge_gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=args.amp and torch.cuda.is_available()):
                m1, m2, m3, m4, edge_pred = model(img, dep)
                seg_losses = torch.stack([
                    structure_loss(m1, gt),
                    structure_loss(m2, gt),
                    structure_loss(m3, gt),
                    structure_loss(m4, gt),
                ])
                loss_seg = seg_losses.mean()
                l_egt = edge_gt_loss(edge_pred, edge_gt)
                l_rel = gt.new_tensor(0.0)
                if args.legacy_control_loss and args.ablation_mode in ('goodr', 'full') and (not args.no_rel_loss) and args.rel_loss_weight > 0:
                    l_rel = tgmm_control_loss(
                        model, gt, dep, edge_gt,
                        depth_thr=args.rel_depth_thr,
                        pos_weight=args.rel_pos_weight,
                        neg_weight=args.rel_neg_weight,
                    )
                loss = loss_seg + 0.05 * l_egt + (args.rel_loss_weight * l_rel if args.legacy_control_loss else 0.0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            running['total'] += float(loss.item())
            running['seg'] += float(loss_seg.item())
            running['egt'] += float(l_egt.item())
            running['rel'] += float(l_rel.item())

            denom = max(1, len(loader))
            pbar.set_postfix(
                total=f"{running['total'] / denom:.4f}",
                seg=f"{running['seg'] / denom:.4f}",
                egt=f"{running['egt'] / denom:.4f}",
                rel=f"{running['rel'] / denom:.4f}",
                lr=f"{scheduler.get_lr():.2e}",
            )

        logger.info(
            'Epoch %d: total=%.4f, seg=%.4f, edge_gt=%.4f, legacy_tgmm_control=%.4f, lr=%.3e',
            epoch,
            running['total'] / len(loader),
            running['seg'] / len(loader),
            running['egt'] / len(loader),
            running['rel'] / len(loader),
            scheduler.get_lr(),
        )

        avg_total_loss = running['total'] / len(loader)
        with open(metrics_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                avg_total_loss,
                running['seg'] / len(loader),
                running['egt'] / len(loader),
                running['rel'] / len(loader),
                scheduler.get_lr(),
            ])

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler_epoch': epoch - 1,
            'config': config_snapshot,
            'ablation_mode': args.ablation_mode,
            'model_kwargs': {'pretrained': False, 'ablation_mode': args.ablation_mode},
        }
        if ema is not None:
            ckpt['ema_model'] = ema.state_dict()
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            ckpt['best_loss'] = best_loss
            ckpt['best_epoch'] = epoch
            atomic_save(ckpt, os.path.join(save_dir, 'best.pth'))
            logger.info('New best.pth by minimum training loss: epoch=%d, loss=%.6f', epoch, best_loss)
        if epoch % cfg.save_interval == 0 or epoch in (175, 180, 200) or epoch == cfg.epochs:
            atomic_save(ckpt, os.path.join(save_dir, f'epoch_{epoch}.pth'))
        atomic_save(ckpt, os.path.join(save_dir, 'latest.pth'))
        scheduler.step()


if __name__ == '__main__':
    main()
