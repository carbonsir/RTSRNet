#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profile RTSRNet / RGB-D COD models.

Put this file in the project root, for example:
    RTSRNet/profile_model.py

It reports:
    1. Total / trainable parameters
    2. Top-level parameter breakdown
    3. MACs / FLOPs if thop is installed
    4. Inference latency and FPS

Recommended:
    pip install thop

Example:
    python profile_model.py --img_size 352 --batch_size 1 --device cuda --amp

With checkpoint:
    python profile_model.py \
      --ckpt results/RTSRNet/best.pth \
      --img_size 352 \
      --batch_size 1 \
      --device cuda \
      --amp

For more stable FPS:
    python profile_model.py --img_size 352 --batch_size 1 --warmup 100 --runs 500 --amp
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def human_num(n: float) -> str:
    if abs(n) >= 1e9:
        return f"{n / 1e9:.4f}G"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.4f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.4f}K"
    return f"{n:.0f}"


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def top_level_param_breakdown(model: torch.nn.Module) -> Dict[str, int]:
    breakdown: Dict[str, int] = {}
    for name, module in model.named_children():
        breakdown[name] = sum(p.numel() for p in module.parameters())
    return dict(sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True))


def named_param_breakdown(model: torch.nn.Module, max_depth: int = 2) -> Dict[str, int]:
    """
    Aggregates parameters by module name up to max_depth.
    Example:
        decoder.tgmm2.xxx -> decoder.tgmm2 when max_depth=2
    """
    out: Dict[str, int] = {}
    for name, p in model.named_parameters():
        parts = name.split(".")[:-1]
        if not parts:
            key = "<root>"
        else:
            key = ".".join(parts[:max_depth])
        out[key] = out.get(key, 0) + p.numel()
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, use_raw: bool = False) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if (not use_raw) and "ema_model" in ckpt:
            state = ckpt["ema_model"]
            state_name = "ema_model"
        elif "model" in ckpt:
            state = ckpt["model"]
            state_name = "model"
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
            state_name = "state_dict"
        else:
            state = ckpt
            state_name = "raw_dict"
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    # Strip common prefixes.
    clean_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        clean_state[nk] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    info = {
        "checkpoint": ckpt_path,
        "state_used": state_name,
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "best_epoch": ckpt.get("best_epoch") if isinstance(ckpt, dict) else None,
        "best_loss": ckpt.get("best_loss") if isinstance(ckpt, dict) else None,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "missing_key_examples": list(missing)[:10],
        "unexpected_key_examples": list(unexpected)[:10],
    }
    return info


@torch.no_grad()
def measure_fps(
    model: torch.nn.Module,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    device: torch.device,
    warmup: int,
    runs: int,
    amp: bool,
    channels_last: bool,
) -> Dict[str, float]:
    model.eval()

    if channels_last:
        rgb = rgb.contiguous(memory_format=torch.channels_last)
        # depth is 1-channel; channels_last is still valid for 4D tensors.
        depth = depth.contiguous(memory_format=torch.channels_last)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        if device.type == "cuda" and amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(rgb, depth)
        else:
            _ = model(rgb, depth)
    sync()

    start = time.perf_counter()
    for _ in range(runs):
        if device.type == "cuda" and amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(rgb, depth)
        else:
            _ = model(rgb, depth)
    sync()
    elapsed = time.perf_counter() - start

    batch_size = rgb.shape[0]
    latency_ms_per_batch = elapsed * 1000.0 / max(runs, 1)
    latency_ms_per_image = latency_ms_per_batch / batch_size
    fps = batch_size * runs / elapsed if elapsed > 0 else float("nan")

    return {
        "runs": runs,
        "warmup": warmup,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "latency_ms_per_batch": latency_ms_per_batch,
        "latency_ms_per_image": latency_ms_per_image,
        "fps_images_per_second": fps,
    }


def profile_macs_with_thop(model: torch.nn.Module, rgb: torch.Tensor, depth: torch.Tensor) -> Dict[str, Any]:
    try:
        from thop import profile, clever_format
    except Exception as e:
        return {
            "available": False,
            "error": repr(e),
            "hint": "Install thop first: pip install thop",
        }

    model.eval()
    # thop returns MACs for the given batch input.
    macs, params_from_thop = profile(model, inputs=(rgb, depth), verbose=False)
    macs_per_image = macs / max(rgb.shape[0], 1)
    flops_per_image_2x = macs_per_image * 2.0

    macs_str, params_str = clever_format([macs_per_image, params_from_thop], "%.4f")
    flops_str, _ = clever_format([flops_per_image_2x, params_from_thop], "%.4f")

    return {
        "available": True,
        "macs_total_for_batch": float(macs),
        "macs_per_image": float(macs_per_image),
        "flops_per_image_2x_macs": float(flops_per_image_2x),
        "macs_per_image_readable": macs_str,
        "flops_per_image_2x_readable": flops_str,
        "params_from_thop": int(params_from_thop),
        "params_from_thop_readable": params_str,
        "note": (
            "THOP reports MACs. Many papers call MACs as FLOPs, while some use FLOPs≈2×MACs. "
            "Report clearly which convention you use."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="Optional checkpoint path.")
    parser.add_argument("--use_raw", action="store_true", help="Use raw model weights instead of ema_model if both exist.")
    parser.add_argument("--img_size", type=int, default=352, help="Input H=W.")
    parser.add_argument("--height", type=int, default=0, help="Override input height.")
    parser.add_argument("--width", type=int, default=0, help="Override input width.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp", action="store_true", help="Use torch autocast fp16 for FPS test on CUDA.")
    parser.add_argument("--channels_last", action="store_true", help="Use channels_last memory format for FPS test.")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--save_json", type=str, default="", help="Optional JSON output path.")
    parser.add_argument("--breakdown_depth", type=int, default=2, help="Depth for detailed parameter aggregation.")
    parser.add_argument("--ablation_mode", type=str, default="full", choices=["baseline", "dgp", "dgp_dls", "full"],
                        help="Profile a specific ablation branch.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Import inside main so the script can show a useful message if placed outside project root.
    try:
        from Model.RTSRNet import RTSRNet
    except Exception as e:
        raise RuntimeError(
            "Could not import Model.RTSRNet. Put profile_model.py in the project root, "
            "then run it from that directory."
        ) from e

    torch.backends.cudnn.benchmark = True

    model = RTSRNet(pretrained=False, ablation_mode=args.ablation_mode)
    ckpt_info = None
    if args.ckpt:
        ckpt_info = load_checkpoint(model, args.ckpt, use_raw=args.use_raw)

    model.to(device)
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    h = args.height if args.height > 0 else args.img_size
    w = args.width if args.width > 0 else args.img_size

    rgb = torch.randn(args.batch_size, 3, h, w, device=device)
    depth = torch.randn(args.batch_size, 1, h, w, device=device)
    if args.channels_last:
        rgb = rgb.contiguous(memory_format=torch.channels_last)
        depth = depth.contiguous(memory_format=torch.channels_last)

    # Basic forward sanity check.
    with torch.no_grad():
        if device.type == "cuda" and args.amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(rgb, depth)
        else:
            outputs = model(rgb, depth)

    output_shapes = [list(o.shape) for o in outputs] if isinstance(outputs, (tuple, list)) else [list(outputs.shape)]

    total_params, trainable_params = count_parameters(model)
    top_breakdown = top_level_param_breakdown(model)
    detailed_breakdown = named_param_breakdown(model, max_depth=args.breakdown_depth)

    # THOP can run on CUDA or CPU. Keep it on the selected device to avoid mismatch.
    macs_info = profile_macs_with_thop(model, rgb, depth)

    fps_info = measure_fps(
        model=model,
        rgb=rgb,
        depth=depth,
        device=device,
        warmup=args.warmup,
        runs=args.runs,
        amp=args.amp,
        channels_last=args.channels_last,
    )

    cuda_info = {}
    if device.type == "cuda":
        cuda_info = {
            "device_name": torch.cuda.get_device_name(device),
            "max_memory_allocated_MB": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
            "max_memory_reserved_MB": torch.cuda.max_memory_reserved(device) / 1024 / 1024,
        }

    result = {
        "model": "RTSRNet",
        "ablation_mode": args.ablation_mode,
        "input": {
            "batch_size": args.batch_size,
            "height": h,
            "width": w,
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
        },
        "device": str(device),
        "amp": bool(args.amp),
        "channels_last": bool(args.channels_last),
        "checkpoint_info": ckpt_info,
        "output_shapes": output_shapes,
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "total_M": total_params / 1e6,
            "trainable_M": trainable_params / 1e6,
            "top_level": top_breakdown,
            "top_level_M": {k: v / 1e6 for k, v in top_breakdown.items()},
            "detailed": detailed_breakdown,
            "detailed_M": {k: v / 1e6 for k, v in detailed_breakdown.items()},
        },
        "complexity": macs_info,
        "speed": fps_info,
        "cuda": cuda_info,
    }

    print("\n================ Model Profile ================")
    print(f"Input: RGB {tuple(rgb.shape)}, Depth {tuple(depth.shape)}")
    print(f"Device: {device}, AMP: {args.amp}, channels_last: {args.channels_last}")
    if ckpt_info:
        print(f"Checkpoint: {ckpt_info['checkpoint']}")
        print(f"State used: {ckpt_info['state_used']}, epoch={ckpt_info['epoch']}, best_epoch={ckpt_info['best_epoch']}")
        print(f"Missing keys: {ckpt_info['missing_keys']}, Unexpected keys: {ckpt_info['unexpected_keys']}")

    print("\n[Parameters]")
    print(f"Total params:     {total_params:,} ({total_params / 1e6:.4f} M)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params / 1e6:.4f} M)")
    print("\nTop-level breakdown:")
    for k, v in top_breakdown.items():
        print(f"  {k:<24s} {v:>12,}  {v / 1e6:>8.4f} M")

    print("\n[Complexity]")
    if macs_info.get("available"):
        print(f"MACs / image:             {macs_info['macs_per_image_readable']} ({macs_info['macs_per_image'] / 1e9:.4f} G)")
        print(f"FLOPs / image, 2x MACs:   {macs_info['flops_per_image_2x_readable']} ({macs_info['flops_per_image_2x_macs'] / 1e9:.4f} G)")
        print("Note: THOP reports MACs. Use a consistent convention when comparing with BPNet.")
    else:
        print("THOP not available, MACs/FLOPs skipped.")
        print(f"Reason: {macs_info.get('error')}")
        print(macs_info.get("hint"))

    print("\n[Speed]")
    print(f"Latency / image: {fps_info['latency_ms_per_image']:.4f} ms")
    print(f"Latency / batch: {fps_info['latency_ms_per_batch']:.4f} ms")
    print(f"FPS:             {fps_info['fps_images_per_second']:.2f} images/s")
    if cuda_info:
        print("\n[CUDA memory]")
        print(f"GPU: {cuda_info['device_name']}")
        print(f"Max allocated: {cuda_info['max_memory_allocated_MB']:.2f} MB")
        print(f"Max reserved:  {cuda_info['max_memory_reserved_MB']:.2f} MB")

    print("================================================\n")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON profile to: {out_path}")


if __name__ == "__main__":
    main()
