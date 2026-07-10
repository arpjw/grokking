"""Full-batch training loop with periodic eval and CSV logging."""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import time

import torch
import torch.nn.functional as F
import yaml

from data import make_data, vocab_size
from model import Transformer


def pick_device(pref: str) -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits = model(x)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(-1) == y).float().mean().item()
    model.train()
    return loss, acc


def train(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["run_name"]
    seed = int(cfg["seed"])
    torch.manual_seed(seed)

    p = int(cfg["task"]["p"])
    frac_train = float(cfg["task"]["frac_train"])
    train_x, train_y, test_x, test_y = make_data(p, frac_train, seed)

    device = pick_device(str(cfg["train"]["device"]))
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    m_cfg = cfg["model"]
    model = Transformer(
        vocab_size=vocab_size(p),
        seq_len=train_x.shape[1],
        n_layers=int(m_cfg["n_layers"]),
        d_model=int(m_cfg["d_model"]),
        n_heads=int(m_cfg["n_heads"]),
        d_mlp=int(m_cfg["d_mlp"]),
    ).to(device)

    o = cfg["optim"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(o["lr"]),
        weight_decay=float(o["weight_decay"]),
        betas=tuple(float(x) for x in o["betas"]),
    )

    out_dir = os.path.join("results", run_name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(out_dir, "config.yaml"))
    csv_path = os.path.join(out_dir, "metrics.csv")

    total_steps = int(cfg["train"]["total_steps"])
    log_every = int(cfg["train"]["log_every"])
    ckpt_every = int(cfg["train"].get("ckpt_every", 0))
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    if ckpt_every > 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"run={run_name} device={device} "
        f"n_train={train_x.shape[0]} n_test={test_x.shape[0]} n_params={n_params}"
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["step", "train_loss", "train_acc", "test_loss", "test_acc", "elapsed_s"]
        )
        t0 = time.time()
        for step in range(total_steps + 1):
            if step % log_every == 0:
                train_loss, train_acc = evaluate(model, train_x, train_y)
                test_loss, test_acc = evaluate(model, test_x, test_y)
                elapsed = time.time() - t0
                writer.writerow(
                    [step, train_loss, train_acc, test_loss, test_acc, elapsed]
                )
                f.flush()
                print(
                    f"step={step:6d} train_loss={train_loss:.4f} "
                    f"train_acc={train_acc:.4f} test_loss={test_loss:.4f} "
                    f"test_acc={test_acc:.4f} t={elapsed:.1f}s"
                )

            if ckpt_every > 0 and step % ckpt_every == 0:
                ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
                torch.save(
                    {"step": step, "model_state_dict": model.state_dict()},
                    ckpt_path,
                )

            if step == total_steps:
                break

            logits = model(train_x)
            loss = F.cross_entropy(logits, train_y)
            opt.zero_grad()
            loss.backward()
            opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
