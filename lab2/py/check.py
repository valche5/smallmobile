"""Vérifie que Rust (lab2/brain/src/net.rs) calcule exactement le même réseau que PyTorch.

    python check.py parties.v6.bin   → écrit /tmp/smallmobile-check.bin, puis compare avec `lab2 check-net`
"""

import os
import re
import subprocess
import sys

import numpy as np
import torch

from data import RECORD, load
from model import Net, export, unpack_adj

torch.manual_seed(0)
net = Net(hidden=32, layers=3)
with torch.no_grad():
    for p in net.parameters():
        p.add_(torch.randn_like(p) * 0.1)  # aucune branche nulle : tout le calcul compte
export(net, "/tmp/smallmobile-check.bin")

recs = load([sys.argv[1]])[0][:5]
node = torch.from_numpy(recs["node"].copy())
glob = torch.from_numpy(recs["glob"].copy())
adj = unpack_adj(torch.from_numpy(recs["adj"].astype(np.int64)))
regions = torch.from_numpy(recs["regions"].astype(np.int64))
with torch.no_grad():
    logits, value, beh = net(node, glob, adj, regions)

out = subprocess.run(
    [os.environ.get("LAB", "../../target/release/lab2"), "check-net", "--net", "/tmp/smallmobile-check.bin", "--data", sys.argv[1]],
    capture_output=True,
    text=True,
    check=True,
).stdout.splitlines()
worst = 0.0
for i, line in enumerate(out):
    n = int(recs["players"][i])
    win = torch.softmax(value[i, :n], 0)
    rust_win = [float(x) for x in re.search(r"win \[([^\]]*)\]", line).group(1).split(",")][:n]
    worst = max(worst, max(abs(a - b) for a, b in zip(win.tolist(), rust_win)))
    policy, _, behaviour = line.split("logits")[1].partition(" behaviour ")
    assert behaviour, "Rust ne lit pas la tête « coup réel »"
    for out, text in ((logits, policy), (beh, behaviour)):
        for k, v in re.findall(r"(\d+):(-?[\d.]+)", text):
            worst = max(worst, abs(out[i, int(k)].item() - float(v)))
print(f"{len(out)} positions, plus grand écart Rust / PyTorch : {worst:.2e}")
assert worst < 1e-3, "Rust et PyTorch ne calculent pas le même réseau"
print("ok, dtype", RECORD.itemsize, "octets par position")
