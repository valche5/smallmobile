"""Le réseau, côté PyTorch. Même architecture que lab2/brain/src/net.rs, qui le recalcule en Rust.

    python model.py ../nets/gen0.bin     écrit un réseau neuf (gen0 : il joue au hasard)
"""

import struct

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from data import GLOBAL, KEYS, NODE, REGIONS

GLOBAL_KEYS = 32
# la tête de valeur prédit les pièces gagnées d'ici la fin divisées par ce nombre
GAIN_SCALE = 50.0
REGION_KINDS = 9
# l'histoire des joueurs (lab2/brain/src/encode.rs) : habitudes de chaque joueur, puis régions prises par
# chacun à chacun, depuis le début et pendant le dernier tour
HABITS = 14
G_HABITS = 700
G_ATTACKS = G_HABITS + 5 * HABITS
G_RECENT = G_ATTACKS + 25


def scales():
    """Échelles des entrées (indices : constantes de lab2/brain/src/encode.rs). Exportées avec les poids."""
    node = np.ones(NODE, np.float32)
    node[13] = 0.1  # éloignement du centre
    node[14] = 0.1  # nombre de voisines
    for i in (58, 65, 68, 69, 73):  # jetons, défense, coût, manque au dé, jetons ramassés
        node[i] = 0.2
    node[59] = 0.5  # campements
    node[76] = 0.25  # voisines au peuple qui conquiert
    glob = np.ones(GLOBAL, np.float32)
    glob[4:7] = 0.1  # manche, manches, manches restantes
    glob[18:20] = 0.2  # conquêtes, régions non vides
    glob[22] = 0.33  # dé des Berserks
    glob[26] = 0.25  # bonus des Amazones
    glob[28] = 0.1  # jetons en main
    glob[63] = 0.5  # défenses en attente
    glob[64:67] = 0.1  # pioches
    P, M = 81, 38
    for q in range(5):
        b = 67 + q * P
        glob[b + 1] = 0.02  # pièces
        glob[b + 2] = 0.1
        glob[b + 3] = 0.1
        glob[b + 38] = 0.1
        glob[b + 39] = 0.1
        glob[b + 40] = 0.05
        glob[b + 75] = 0.1
    for m in range(6):
        b = 67 + 5 * P + m * M
        glob[b + 35] = 0.2
        glob[b + 36] = 0.1
    glob[G_HABITS:G_ATTACKS] = 0.1  # comptes de chaque joueur (lab2/brain/src/know.rs, `habit`)
    for q in range(5):
        glob[G_HABITS + q * HABITS + 13] = 0.05  # jetons perdus sous les attaques
    glob[G_ATTACKS:G_RECENT] = 0.1  # régions prises depuis le début
    glob[G_RECENT : G_RECENT + 25] = 0.25  # pendant le dernier tour
    return node, glob


class Layer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.n1 = nn.Linear(3 * h, h)
        self.n2 = nn.Linear(h, h)
        self.g1 = nn.Linear(3 * h, h)
        self.g2 = nn.Linear(h, h)
        # les branches résiduelles partent de zéro : le réseau commence simple
        for lin in (self.n2, self.g2):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)


class Net(nn.Module):
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        h = self.hidden = hidden
        self.node_in = nn.Linear(NODE, h)
        self.glob_in = nn.Linear(GLOBAL, h)
        self.layers = nn.ModuleList(Layer(h) for _ in range(layers))
        self.node_pol = nn.Linear(2 * h, REGION_KINDS)
        self.glob_pol1 = nn.Linear(h, h)
        self.glob_pol2 = nn.Linear(h, GLOBAL_KEYS)
        self.val1 = nn.Linear(2 * h, h)
        self.val2 = nn.Linear(h, 10)
        # les sorties partent de zéro : un réseau neuf joue au hasard (politique uniforme, mêmes
        # chances pour tous les joueurs)
        for lin in (self.node_pol, self.glob_pol2, self.val2):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        node, glob = scales()
        self.register_buffer("node_scale", torch.from_numpy(node), persistent=False)
        self.register_buffer("global_scale", torch.from_numpy(glob), persistent=False)

    def forward(self, node, glob, adj, regions):
        """node (B, R, NODE) u8 ; glob (B, GLOBAL) u8 ; adj (B, R, R) 0/1 ; regions (B,) nombre de régions.

        Renvoie les logits des coups (B, KEYS) et la tête de valeur (B, 10)."""
        B, R, _ = node.shape
        mask = (torch.arange(R, device=node.device)[None, :] < regions[:, None]).float()  # (B, R)
        h = self.node_in(node.float() * self.node_scale) * mask[..., None]
        g = self.glob_in(glob.float() * self.global_scale)
        deg = adj.sum(-1).clamp(min=1)[..., None]
        count = mask.sum(1, keepdim=True)
        neg = (1 - mask[..., None]) * -1e9
        for layer in self.layers:
            agg = torch.bmm(adj, h) / deg
            u = torch.cat([h, agg, g[:, None].expand(B, R, -1)], -1)
            h = (h + layer.n2(F.relu(layer.n1(u)))) * mask[..., None]
            mean = h.sum(1) / count
            mx = (h + neg).max(1).values
            g = g + layer.g2(F.relu(layer.g1(torch.cat([g, mean, mx], -1))))
        hg = torch.cat([h, g[:, None].expand(B, R, -1)], -1)

        logits = torch.full((B, KEYS), -1e9, device=node.device)
        logits[:, :GLOBAL_KEYS] = self.glob_pol2(F.relu(self.glob_pol1(g)))
        per_region = self.node_pol(hg)  # (B, R, REGION_KINDS)
        for kind in range(REGION_KINDS):
            start = GLOBAL_KEYS + kind * 64
            logits[:, start : start + R] = per_region[:, :, kind]
        mean = h.sum(1) / count
        value = self.val2(F.relu(self.val1(torch.cat([g, mean], -1))))
        return logits, value


def unpack_adj(bits, R=REGIONS):
    """(B, R) u64 → (B, R, R) 0/1."""
    b = bits.view(torch.int64)
    shifts = torch.arange(R, device=bits.device, dtype=torch.int64)
    return ((b[..., None] >> shifts) & 1).float()


def load(path) -> Net:
    """Un réseau écrit par `export` (format SMNN1) ; largeur et nombre de couches lus dans le fichier."""
    tensors = {}
    with open(path, "rb") as f:
        assert f.read(5) == b"SMNN1", f"{path} : pas un fichier de poids SMNN1"
        (count,) = struct.unpack("<I", f.read(4))
        for _ in range(count):
            (size,) = struct.unpack("<I", f.read(4))
            name = f.read(size).decode()
            (ndim,) = struct.unpack("<I", f.read(4))
            shape = struct.unpack(f"<{ndim}I", f.read(4 * ndim))
            data = np.frombuffer(f.read(4 * int(np.prod(shape))), dtype="<f4").reshape(shape)
            tensors[name] = torch.from_numpy(data.copy())
    layers = sum(1 for k in tensors if k.endswith(".n1.weight"))
    net = Net(tensors["node_in.weight"].shape[0], layers)
    # les échelles des entrées viennent de `scales` : celles du fichier doivent être les mêmes
    for name, scale in (("node_scale", net.node_scale), ("global_scale", net.global_scale)):
        assert torch.equal(tensors.pop(name), scale), f"{path} : échelles des entrées différentes ({name})"
    net.load_state_dict(tensors)
    return net


def export(net: Net, path):
    """Écrit les poids au format SMNN1 lu par lab2/brain/src/net.rs."""
    tensors = [(k, v.detach().float().cpu().contiguous()) for k, v in net.state_dict().items()]
    # les échelles des entrées font partie du modèle : Rust les lit d'ici
    tensors += [("node_scale", net.node_scale.cpu()), ("global_scale", net.global_scale.cpu())]
    with open(path, "wb") as f:
        f.write(b"SMNN1")
        f.write(struct.pack("<I", len(tensors)))
        for name, t in tensors:
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", t.dim()))
            for d in t.shape:
                f.write(struct.pack("<I", d))
            f.write(t.numpy().astype("<f4").tobytes())


if __name__ == "__main__":
    import sys

    torch.manual_seed(0)
    export(Net(), sys.argv[1])
