import argparse
import json
import os
import subprocess
import sys

import cv2
from tqdm import tqdm

from utils.config import Config
from utils.metrics import EvaluationMetricsV2
from Model.RTSRNet import VALID_ABLATION_MODES


DEFAULT_DATASETS = ['CAMO', 'COD10K', 'NC4K', 'CHAMELEON']
VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def find_gt_mask(mask_root, pred_name):
    base = os.path.splitext(pred_name)[0]
    for ext in VALID_EXTS:
        candidate = os.path.join(mask_root, base + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def evaluate_dataset(pred_root, mask_root):
    metric = EvaluationMetricsV2()
    pred_names = sorted([f for f in os.listdir(pred_root) if f.lower().endswith(VALID_EXTS)])
    if not pred_names:
        raise RuntimeError(f'No prediction files found in: {pred_root}')

    for name in tqdm(pred_names, desc=f'Eval {os.path.basename(pred_root)}'):
        pred_path = os.path.join(pred_root, name)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            raise FileNotFoundError(f'Cannot read prediction: {pred_path}')

        mask_path = find_gt_mask(mask_root, name)
        if mask_path is None:
            raise FileNotFoundError(
                f'Cannot find GT mask for {name} in {mask_root}. Tried basename with extensions: {VALID_EXTS}'
            )

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Cannot read GT mask: {mask_path}')

        if pred.shape != mask.shape:
            pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        metric.step(pred=pred, gt=mask)
    return metric.get_results()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_root', type=str, default='')
    p.add_argument('--ckpt', type=str, default='')
    p.add_argument('--run_dir', type=str, default='')
    p.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    p.add_argument('--dataset_dir', type=str, default='')
    p.add_argument('--test_size', type=int, default=None)
    p.add_argument('--save_dir', type=str, default=None)
    p.add_argument('--use_raw', action='store_true', default=False)
    p.add_argument('--save_json', action='store_true', default=False)
    p.add_argument('--json_name', type=str, default='eval_metrics.json')
    p.add_argument('--ablation_mode', type=str, default='', choices=('',) + VALID_ABLATION_MODES,
                   help='Optional inference-time ablation override when --ckpt triggers inference.')
    return p.parse_args()


def resolve_pred_root(args):
    if args.pred_root:
        return args.pred_root
    if args.save_dir:
        return args.save_dir
    if args.ckpt:
        return os.path.join(os.path.dirname(args.ckpt), 'prediction_maps')
    if args.run_dir:
        return os.path.join(args.run_dir, 'prediction_maps')
    raise ValueError('Please provide at least one of --pred_root, --ckpt, or --run_dir')


def ensure_predictions(args):
    pred_root = resolve_pred_root(args)
    need_infer = bool(args.ckpt)
    if need_infer:
        for dataset in args.datasets:
            pred_dir = os.path.join(pred_root, dataset)
            has_preds = os.path.isdir(pred_dir) and any(name.lower().endswith(VALID_EXTS) for name in os.listdir(pred_dir))
            if not has_preds:
                cmd = [sys.executable, 'inference.py', '--ckpt', args.ckpt]
                if args.datasets:
                    cmd += ['--datasets'] + list(args.datasets)
                if args.dataset_dir:
                    cmd += ['--dataset_dir', args.dataset_dir]
                if args.test_size is not None:
                    cmd += ['--test_size', str(args.test_size)]
                if args.save_dir is not None:
                    cmd += ['--save_dir', args.save_dir]
                if args.use_raw:
                    cmd += ['--use_raw']
                if args.ablation_mode:
                    cmd += ['--ablation_mode', args.ablation_mode]
                print('Prediction maps not found. Running inference first:')
                print(' '.join(cmd))
                subprocess.run(cmd, check=True)
                break
    return pred_root


def main():
    args = parse_args()
    cfg = Config()
    if args.dataset_dir:
        cfg.dp.dataset_dir = args.dataset_dir
        cfg.dp.refresh()

    pred_root = ensure_predictions(args)
    summary = {}
    for dataset in args.datasets:
        mask_root = getattr(cfg.dp, f'test_{dataset}_masks')
        pred_dir = os.path.join(pred_root, dataset)
        results = evaluate_dataset(pred_dir, mask_root)
        print(f'\n##### {dataset} #####')
        keep_keys = ['sm', 'emAdp', 'wfm', 'mae']
        compact = {k: results[k] for k in keep_keys}
        summary[dataset] = compact
        print(f"{dataset} {compact['sm']:.3f} {compact['emAdp']:.3f} {compact['wfm']:.3f} {compact['mae']:.3f}")

    if args.save_json:
        out = os.path.join(pred_root, args.json_name)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f'Saved json to: {out}')


if __name__ == '__main__':
    main()
