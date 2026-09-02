#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTSRNet one-click pipeline:
  train -> inference -> evaluation -> analysis

Default mode:
  full = paper-facing RTSRNet configuration

Run:
  python go.py --data /root/autodl-tmp/data
  bash go.sh /root/autodl-tmp/data
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


DEFAULT_DATASETS = ["CAMO", "COD10K", "NC4K", "CHAMELEON"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def q(path) -> str:
    return str(path).replace("\\", "/")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_images(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return -1
    return sum(1 for f in path.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def print_stage_banner(idx: int, total: int, name: str) -> None:
    print("\n" + "=" * 100, flush=True)
    print(f"[{idx}/{total}] {name}", flush=True)
    print("=" * 100, flush=True)


def print_dataset_info(data_root: Path, datasets: List[str]) -> None:
    print("\n[DATA] Using dataset root exactly as provided:", flush=True)
    print("       ", q(data_root), flush=True)
    print("[DATA] Quick check, only for display; it will not block training.", flush=True)

    items = [
        ("Train Imgs", data_root / "TrainDataset" / "Imgs"),
        ("Train GT", data_root / "TrainDataset" / "GT"),
        ("Train Depth", data_root / "TrainDataset" / "Depth"),
    ]
    for ds in datasets:
        items.extend([
            (f"{ds} Imgs", data_root / "TestDataset" / ds / "Imgs"),
            (f"{ds} GT", data_root / "TestDataset" / ds / "GT"),
            (f"{ds} Depth", data_root / "TestDataset" / ds / "Depth"),
        ])

    for name, path in items:
        print(
            f"  {name:16s} exists={str(path.exists()):5s} "
            f"is_dir={str(path.is_dir()):5s} images={count_images(path):5d}  {q(path)}",
            flush=True,
        )
    print("", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTSRNet one-click train/infer/eval/analyze pipeline.")
    p.add_argument("--data", type=str, default="/root/autodl-tmp/data")
    p.add_argument("--out", type=str, default="out")
    p.add_argument("--mode", type=str, default="full",
                   choices=["baseline", "dgp", "dgp_dls", "idls", "goodr", "full"])
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)

    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--bs", "--batch_size", dest="batch_size", type=int, default=16)
    p.add_argument("--size", "--trainsize", dest="size", type=int, default=384)
    p.add_argument("--workers", "--num_workers", dest="workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=False)

    p.add_argument("--rel_loss_weight", type=float, default=0.05)
    p.add_argument("--rel_depth_thr", type=float, default=0.25)
    p.add_argument("--rel_pos_weight", type=float, default=0.5)
    p.add_argument("--rel_neg_weight", type=float, default=2.0)
    p.add_argument("--no_rel_loss", action="store_true", default=False)

    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--raw", action="store_true", default=False)

    p.add_argument("--diag_samples", type=int, default=200, help="Use -1 for full diagnostics.")
    p.add_argument("--save_maps", action="store_true", default=False)
    p.add_argument("--tol", "--tolerance_radius", dest="tol", type=int, default=3)

    p.add_argument("--no_train", action="store_true", default=False)
    p.add_argument("--no_infer", action="store_true", default=False)
    p.add_argument("--no_eval", action="store_true", default=False)
    p.add_argument("--no_ana", action="store_true", default=False)
    p.add_argument("--quiet", action="store_true", default=False)
    return p.parse_args()


def run_cmd(stage_idx: int, stage_total: int, stage_name: str,
            cmd: List[str], cwd: Path, log_path: Path,
            history: List[Dict[str, Any]], quiet: bool = False) -> None:
    print_stage_banner(stage_idx, stage_total, stage_name)
    ensure_dir(log_path.parent)
    print("[CMD]", " ".join(cmd), flush=True)
    print("[LOG]", q(log_path), flush=True)

    start = time.time()
    item = {
        "stage": stage_name,
        "cmd": cmd,
        "cwd": q(cwd),
        "log": q(log_path),
        "start": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
    }

    with log_path.open("a", encoding="utf-8", buffering=1) as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"[{now()}] STAGE: {stage_name}\n")
        f.write(f"[{now()}] RUN: {' '.join(cmd)}\n")
        f.write("=" * 100 + "\n")
        f.flush()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None

        last_tick = time.time()
        for line in proc.stdout:
            f.write(line)
            f.flush()
            if not quiet:
                print(line, end="", flush=True)
            if time.time() - last_tick > 60:
                msg = f"\n[HEARTBEAT] {stage_name} running, elapsed={format_seconds(time.time() - start)}, log={q(log_path)}\n"
                f.write(msg)
                f.flush()
                if not quiet:
                    print(msg, end="", flush=True)
                last_tick = time.time()

        ret = proc.wait()
        elapsed = format_seconds(time.time() - start)
        if ret != 0:
            item["status"] = "failed"
            item["returncode"] = ret
            item["elapsed"] = elapsed
            history.append(item)
            print(f"\n[FAILED] {stage_name}, elapsed={elapsed}", flush=True)
            print("[LOG]", q(log_path), flush=True)
            raise subprocess.CalledProcessError(ret, cmd)

        item["status"] = "ok"
        item["elapsed"] = elapsed
        item["end"] = datetime.now().isoformat(timespec="seconds")
        history.append(item)

    print(f"\n[OK] {stage_name} finished, elapsed={elapsed}", flush=True)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_train_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    def sf(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    vals = [sf(r.get("total", "")) for r in rows]
    best_i = min(range(len(vals)), key=lambda i: vals[i])
    return {
        "logged_epochs": len(rows),
        "best_train_epoch": rows[best_i].get("epoch", ""),
        "best_train_loss": vals[best_i],
        "final_train_loss": vals[-1],
    }


def write_summary(out_dir: Path, run_dir: Path, pred_dir: Path, ana_dir: Path,
                  args: argparse.Namespace, ckpt: Path, history: List[Dict[str, Any]]) -> None:
    eval_json = pred_dir / "eval_metrics.json"
    train_csv = run_dir / "train_epoch_metrics.csv"
    summary_md = out_dir / "summary.md"
    summary_json = out_dir / "summary.json"

    payload = {
        "time": now(),
        "mode": args.mode,
        "data": args.data,
        "out": q(out_dir),
        "checkpoint": q(ckpt),
        "history": history,
    }

    lines = []
    lines.append("# GoodR summary\n")
    lines.append(f"- Time: `{now()}`")
    lines.append(f"- Mode: `{args.mode}`")
    lines.append("- Architecture: `RTSRNet with SGDEM and TGMM`")
    lines.append(f"- Data: `{args.data}`")
    lines.append(f"- Output: `{q(out_dir)}`")
    lines.append(f"- Checkpoint: `{q(ckpt)}`")
    lines.append("")

    train_info = read_train_csv(train_csv)
    payload["train"] = train_info
    if train_info:
        lines.append("## Training")
        lines.append(f"- Logged epochs: `{train_info['logged_epochs']}`")
        lines.append(f"- Best train epoch: `{train_info['best_train_epoch']}`")
        lines.append(f"- Best train loss: `{train_info['best_train_loss']:.6f}`")
        lines.append(f"- Final train loss: `{train_info['final_train_loss']:.6f}`")
        lines.append("")

    if eval_json.exists():
        ev = read_json(eval_json)
        payload["eval"] = ev
        lines.append("## Evaluation")
        lines.append("| Dataset | Sm | EmAdp | wFm | MAE |")
        lines.append("|---|---:|---:|---:|---:|")
        for ds in args.datasets:
            r = ev.get(ds, {})
            lines.append(
                f"| {ds} | {float(r.get('sm', 0)):.4f} | "
                f"{float(r.get('emAdp', 0)):.4f} | "
                f"{float(r.get('wfm', 0)):.4f} | "
                f"{float(r.get('mae', 0)):.4f} |"
            )
        lines.append("")

    lines.append("## Files")
    lines.append(f"- Train log: `{q(out_dir / 'log' / 'train.log')}`")
    lines.append(f"- Inference log: `{q(out_dir / 'log' / 'infer.log')}`")
    lines.append(f"- Evaluation log: `{q(out_dir / 'log' / 'eval.log')}`")
    lines.append(f"- Analysis log: `{q(out_dir / 'log' / 'ana.log')}`")
    lines.append(f"- Prediction maps: `{q(pred_dir)}`")
    lines.append(f"- Diagnostics: `{q(ana_dir)}`")
    lines.append(f"- Evaluation JSON: `{q(eval_json)}`")

    summary_md.write_text("\n".join(lines), encoding="utf-8")
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100, flush=True)
    print("[ALL DONE]", flush=True)
    print("Summary:", q(summary_md), flush=True)
    print("JSON:", q(summary_json), flush=True)
    print("=" * 100, flush=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_root = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out).expanduser()
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir = out_dir.resolve()

    run_dir = out_dir / "run"
    pred_dir = out_dir / "pred"
    ana_dir = out_dir / "ana"
    log_dir = out_dir / "log"
    for d in [run_dir, pred_dir, ana_dir, log_dir]:
        ensure_dir(d)

    print_dataset_info(data_root, args.datasets)

    py = sys.executable
    history: List[Dict[str, Any]] = []
    total_steps = sum([not args.no_train, not args.no_infer, not args.no_eval, not args.no_ana])
    step = 0

    if not args.no_train:
        step += 1
        cmd = [
            py, "-u", "train.py",
            "--ablation_mode", args.mode,
            "--dataset_dir", q(data_root),
            "--save_dir", q(run_dir),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--trainsize", str(args.size),
            "--num_workers", str(args.workers),
            "--seed", str(args.seed),
            "--rel_loss_weight", str(args.rel_loss_weight),
            "--rel_depth_thr", str(args.rel_depth_thr),
            "--rel_pos_weight", str(args.rel_pos_weight),
            "--rel_neg_weight", str(args.rel_neg_weight),
        ]
        if args.no_rel_loss:
            cmd.append("--no_rel_loss")
        if args.amp:
            cmd.append("--amp")
        run_cmd(step, total_steps, "Training", cmd, root, log_dir / "train.log", history, quiet=args.quiet)

    ckpt = Path(args.ckpt).expanduser().resolve() if args.ckpt else run_dir / "best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {q(ckpt)}")

    if not args.no_infer:
        step += 1
        cmd = [
            py, "-u", "inference.py",
            "--ckpt", q(ckpt),
            "--datasets", *args.datasets,
            "--dataset_dir", q(data_root),
            "--test_size", str(args.size),
            "--save_dir", q(pred_dir),
            "--ablation_mode", args.mode,
        ]
        if args.raw:
            cmd.append("--use_raw")
        run_cmd(step, total_steps, "Inference", cmd, root, log_dir / "infer.log", history, quiet=args.quiet)

    if not args.no_eval:
        step += 1
        cmd = [
            py, "-u", "evaluate.py",
            "--pred_root", q(pred_dir),
            "--datasets", *args.datasets,
            "--dataset_dir", q(data_root),
            "--save_json",
            "--json_name", "eval_metrics.json",
        ]
        run_cmd(step, total_steps, "Evaluation", cmd, root, log_dir / "eval.log", history, quiet=args.quiet)

    if not args.no_ana:
        step += 1
        cmd = [
            py, "-u", "analyze_bridge_diagnostics.py",
            "--checkpoint", q(ckpt),
            "--data_root", q(data_root),
            "--datasets", ",".join(args.datasets),
            "--save_root", q(ana_dir),
            "--image_size", str(args.size),
            "--tolerance_radius", str(args.tol),
            "--ablation_mode", args.mode,
        ]
        if args.diag_samples is not None and args.diag_samples >= 0:
            cmd += ["--max_samples", str(args.diag_samples)]
        if args.save_maps:
            cmd.append("--save_maps")
        if args.raw:
            cmd.append("--use_raw")
        run_cmd(step, total_steps, "Analysis", cmd, root, log_dir / "ana.log", history, quiet=args.quiet)

    write_summary(out_dir, run_dir, pred_dir, ana_dir, args, ckpt, history)


if __name__ == "__main__":
    main()
