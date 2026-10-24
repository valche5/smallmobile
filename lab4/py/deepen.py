"""Approfondit un réseau : ses couches, puis `--add` couches neuves dont les branches résiduelles
partent de zéro (`model.Layer`). Le réseau approfondi calcule exactement la même chose que l'ancien,
et peut continuer à apprendre avec plus de capacité (pistes pour sortir du plateau, lab4/journal.md).

    python deepen.py --init ../runs/p2-3-4-5/c015.bin --add 2 --out ../runs/deep6/c016.bin
"""

import argparse

import torch

from data import GLOBAL, NODE, REGIONS
from model import Net, export, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="le réseau à approfondir (.bin)")
    ap.add_argument("--add", type=int, default=2, help="couches ajoutées à la fin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    net = load(args.init)
    deeper = Net(net.hidden, len(net.layers) + args.add)
    missing, unexpected = deeper.load_state_dict(net.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("layers.") for k in missing), (missing, unexpected)
    # mêmes sorties, sur des entrées au hasard
    B = 8
    node = torch.randint(0, 3, (B, REGIONS, NODE), dtype=torch.uint8)
    glob = torch.randint(0, 3, (B, GLOBAL), dtype=torch.uint8)
    adj = (torch.rand(B, REGIONS, REGIONS) < 0.1).float()
    regions = torch.randint(10, REGIONS + 1, (B,))
    with torch.no_grad():
        for a, b in zip(net(node, glob, adj, regions), deeper(node, glob, adj, regions)):
            assert torch.equal(a, b), "le réseau approfondi ne calcule pas la même chose"
    export(deeper, args.out)
    print(f"{args.init} ({len(net.layers)} couches) → {args.out} ({len(deeper.layers)} couches), mêmes sorties")


if __name__ == "__main__":
    main()
