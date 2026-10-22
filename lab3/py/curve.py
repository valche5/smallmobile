"""Plusieurs réseaux mesurés sur les mêmes positions de validation : une courbe comparable d'un
réseau à l'autre (les mesures de train.py, elles, changent de parties avec la fenêtre).

    python curve.py --data ../data/p4/c019 ../data/p4/c020 --nets ../runs/p4/c010.bin ../runs/p4/c020.bin

Positions : les parties de validation (`in_validation`) des cycles donnés, jamais vues à
l'entraînement par aucun réseau de la boucle.
"""

import argparse
from pathlib import Path

import numpy as np

from data import in_validation, load as load_data
from model import load as load_net
from train import VAL_SHARE, show, validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--nets", nargs="+", required=True)
    args = ap.parse_args()
    recs, keys = load_data(args.data, keep=lambda k: in_validation(k, VAL_SHARE))
    idx = np.arange(len(recs))
    print(f"{len(recs)} positions de validation, {len(np.unique(keys))} parties", flush=True)
    for path in args.nets:
        net = load_net(path)
        print(f"{Path(path).stem:12} {net.hidden}x{len(net.layers)} : {show(validate(net, recs, idx, 'cpu'))}", flush=True)


if __name__ == "__main__":
    main()
