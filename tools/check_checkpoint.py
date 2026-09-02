#!/usr/bin/env python3
"""Validate that an RTSRNet or legacy GoodRT_v2 checkpoint loads strictly."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from Model.RTSRNet import RTSRNet, extract_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--use_raw", action="store_true",
                        help="Prefer checkpoint['model'] over checkpoint['ema_model'].")
    parser.add_argument("--ablation_mode", default="",
                        help="Override checkpoint mode; default is read from checkpoint or 'full'.")
    parser.add_argument("--img_size", type=int, default=64,
                        help="Optional random-input smoke-test size; use 0 to skip.")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    mode = args.ablation_mode
    if not mode and isinstance(checkpoint, dict):
        mode = checkpoint.get("ablation_mode", "")
        if not mode:
            mode = checkpoint.get("config", {}).get("ablation_mode", "")
    mode = mode or "full"

    state = extract_state_dict(checkpoint, use_ema=not args.use_raw)
    model = RTSRNet(pretrained=False, ablation_mode=mode).eval()
    result = model.load_state_dict(state, strict=True)
    print(f"[OK] strict checkpoint loading succeeded: {result}")
    print(f"[OK] ablation_mode={mode}, parameters={sum(p.numel() for p in model.parameters()):,}")

    if args.img_size > 0:
        rgb = torch.randn(1, 3, args.img_size, args.img_size)
        depth = torch.randn(1, 1, args.img_size, args.img_size)
        with torch.no_grad():
            outputs = model(rgb, depth)
        print("[OK] forward smoke test:")
        for i, out in enumerate(outputs, 1):
            print(f"  output_{i}: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
