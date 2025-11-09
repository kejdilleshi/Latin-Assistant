#!/usr/bin/env python
import os
# Force a non-interactive backend if no DISPLAY (headless environments)
import matplotlib
matplotlib.use("Agg")  # force headless backend on clusters
import matplotlib.pyplot as plt
import argparse
import json
import math
from typing import Dict, Any, List

import matplotlib.pyplot as plt


def load_trainer_state(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_series(log_history: List[Dict[str, Any]]):
    steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for rec in log_history:
        # HF Trainer logs; keys vary across versions
        if "loss" in rec and "step" in rec:
            steps.append(rec["step"])
            train_loss.append(rec["loss"])
        if "eval_loss" in rec and "step" in rec:
            eval_steps.append(rec["step"])
            eval_loss.append(rec["eval_loss"])
    return (steps, train_loss), (eval_steps, eval_loss)


def moving_avg(vals: List[float], k: int = 5) -> List[float]:
    if k <= 1 or len(vals) == 0:
        return vals
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - k + 1), i + 1
        out.append(sum(vals[lo:hi]) / (hi - lo))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trainer_state", type=str,
                    help="Path to trainer_state.json (e.g., results/.../trainer_state.json)")
    ap.add_argument("--smooth", type=int, default=5,
                    help="Moving-average window; set 1 to disable")
    ap.add_argument("--out", type=str, default="loss_curve.png",
                    help="Output PNG filename")
    args = ap.parse_args()

    state = load_trainer_state(args.trainer_state)
    log_history = state.get("log_history", [])

    (s_steps, s_train), (e_steps, s_eval) = extract_series(log_history)
    if not s_steps and not e_steps:
        raise RuntimeError("Couldn't find 'loss' or 'eval_loss' entries in log_history.")

    # Optional smoothing
    s_train_s = moving_avg(s_train, args.smooth)
    s_eval_s = moving_avg(s_eval, max(1, min(args.smooth, 3)))  # lighter smoothing for eval

    plt.figure(figsize=(9, 5))
    if s_steps:
        plt.plot(s_steps, s_train_s, label="train loss")
    if e_steps:
        plt.plot(e_steps, s_eval_s, label="eval loss")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training/Eval Loss vs Steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"[OK] Saved plot to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
