"""Le réseau sur GPU pour la génération des parties (`lab3/src/gpu.rs`, option `--gpu` de
`lab3 selfplay`) : le moteur Rust lui passe des lots de positions et reprend logits et valeurs.

Appelé depuis Rust (pyo3) : `Evaluator(chemin des poids)`, puis
`evaluate(count, node, glob, adj, regions)` avec des octets au format des enregistrements
(`lab3/src/selfplay.rs`) : node (count × REGIONS × NODE u8), glob (count × GLOBAL u8), adj
(count × REGIONS u64, un bit par voisine), regions (count u8). Rend les logits des coups
(count × KEYS f32) et la tête de valeur (count × 10 f32), en octets ; Rust en tire victoire et
pièces comme de son propre calcul (`net::Eval::from_heads`).

Calcul en bf16 sur GPU (CUDA ou ROCm, tous deux `torch.cuda`), en f32 sur processeur (essais sur
mini).
"""

import torch

from data import GLOBAL, NODE, REGIONS
from model import load, unpack_adj


class Evaluator:
    def __init__(self, path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = load(path).to(self.device).eval()

    def describe(self):
        if self.device == "cuda":
            return f"GPU {torch.cuda.get_device_name()} (bf16)"
        return f"processeur, {torch.get_num_threads()} fils (f32)"

    @torch.no_grad()
    def evaluate(self, count, node, glob, adj, regions):
        d = self.device
        node = torch.frombuffer(node, dtype=torch.uint8).view(count, REGIONS, NODE).to(d, non_blocking=True)
        glob = torch.frombuffer(glob, dtype=torch.uint8).view(count, GLOBAL).to(d, non_blocking=True)
        adj = torch.frombuffer(adj, dtype=torch.int64).view(count, REGIONS).to(d, non_blocking=True)
        regions = torch.frombuffer(regions, dtype=torch.uint8).to(d, non_blocking=True).long()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=d == "cuda"):
            logits, value, _ = self.net(node, glob, unpack_adj(adj), regions)
        return logits.float().cpu().numpy().tobytes(), value.float().cpu().numpy().tobytes()
