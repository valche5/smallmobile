"""Débit du réseau sur CUDA (PyTorch), à comparer à `lab6 bench-vulkan` sur la même carte.

    python bench_cuda.py ../nets/c130.bin [--regions 40] [--seconds 1]

Positions de forme réaliste (40 régions sur 48, environ 5 voisines chacune, entrées entières
petites) ; le réseau sans la tête des propriétaires, que Rust ne calcule pas. Trois modes : f32,
bf16 (autocast) et bf16 compilé (`torch.compile`, graphes CUDA)."""

import argparse
import time

import torch
from torch import nn

from data import GLOBAL, NODE, REGIONS
from model import load

ap = argparse.ArgumentParser()
ap.add_argument("net")
ap.add_argument("--regions", type=int, default=40)
ap.add_argument("--seconds", type=float, default=1.0)
args = ap.parse_args()

dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
net = load(args.net)
net.own = nn.Linear(2 * net.hidden, 0)  # Rust ne calcule pas cette tête
net = net.to(dev).eval()
print(f"{torch.cuda.get_device_name()}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"réseau {net.hidden} × {len(net.layers)}, {args.regions} régions par position")


def inputs(batch):
    g = torch.Generator(device="cpu").manual_seed(batch)
    node = torch.randint(0, 4, (batch, REGIONS, NODE), dtype=torch.uint8, generator=g)
    glob = torch.randint(0, 4, (batch, GLOBAL), dtype=torch.uint8, generator=g)
    adj = (torch.rand(batch, REGIONS, REGIONS, generator=g) < 5 / args.regions).float()
    adj = ((adj + adj.transpose(1, 2)) > 0).float()
    adj[:, args.regions :, :] = 0
    adj[:, :, args.regions :] = 0
    regions = torch.full((batch,), args.regions, dtype=torch.int64)
    return [t.to(dev) for t in (node, glob, adj, regions)]


def rate(fn, x):
    with torch.no_grad():
        for _ in range(5):
            fn(*x)
        torch.cuda.synchronize()
        n, t = 0, time.perf_counter()
        while time.perf_counter() - t < args.seconds:
            fn(*x)
            n += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
    return n * x[0].shape[0] / dt, dt / n * 1e3


def bf16(*x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return net(*x)


compiled = torch.compile(bf16, mode="max-autotune-no-cudagraphs", dynamic=False)
graphed = torch.compile(bf16, mode="reduce-overhead", dynamic=False)
print(f"{'lot':>6} {'f32':>12} {'bf16':>12} {'bf16 compilé':>14} {'+ graphes':>12}   positions/s (ms par lot)")
for batch in (64, 256, 1024, 2048, 4096):
    x = inputs(batch)
    cols = []
    for fn in (net, bf16, compiled, graphed):
        r, ms = rate(fn, x)
        cols.append(f"{r:>9.0f} ({ms:.2f})")
    print(f"{batch:>6} " + " ".join(cols), flush=True)
