"""Des têtes construites pour le choix du peuple et le déclin, sur le tronc d'un réseau (lab13/lab.md,
D13). Python seul : les moteurs ne les calculent pas.

Dans `model.Net`, ces deux décisions sortent du seul vecteur global `g`. Ici, des corrections ajoutées
à leurs logits, parties de zéro (au départ, le réseau de base exactement) :

- peuple (clés 0 à 5, une par place du marché) : chaque carte résumée par un même petit réseau pour
  les six places ; chaque paire carte × région passée par une même couche, regroupée sur les régions ;
  un même score par carte (Deep Sets) ;
- déclin (clé 7) : une même couche sur chaque région, regroupée sur mes régions et sur toutes.
"""

import torch
from torch import nn
import torch.nn.functional as F

from model import GLOBAL_KEYS, KEYS, REGION_KINDS, joined

# lab13/brain/src/encode.rs : le marché (6 places de M entrées), propriétaire et peuple actif des régions
G_MARKET = 67 + 5 * 81
M = 38
SLOTS = 6
N_OWNER, N_ACTIVE = 17, 22
DECLINE = 7


def trunk(net, node, glob, adj, regions):
    """`model.Net.forward`, qui rend aussi les états des régions `h`, le vecteur global `g` et le masque."""
    B, R, _ = node.shape
    mask = (torch.arange(R, device=node.device)[None, :] < regions[:, None]).float()
    h = net.node_in(node.float() * net.node_scale) * mask[..., None]
    g = net.glob_in(glob.float() * net.global_scale)
    deg = adj.sum(-1).clamp(min=1)[..., None]
    count = mask.sum(1, keepdim=True)
    neg = (1 - mask[..., None]) * -1e9
    for layer in net.layers:
        agg = torch.bmm(adj, h) / deg
        h = (h + layer.n2(F.relu(joined(layer.n1, [h, agg], g)))) * mask[..., None]
        mean = h.sum(1) / count
        mx = (h + neg).max(1).values
        g = g + layer.g2(F.relu(layer.g1(torch.cat([g, mean, mx], -1))))
    logits = torch.full((B, KEYS), -1e9, device=node.device, dtype=h.dtype)
    logits[:, :GLOBAL_KEYS] = net.glob_pol2(F.relu(net.glob_pol1(g)))
    per_region = joined(net.node_pol, [h], g)
    for kind in range(REGION_KINDS):
        start = GLOBAL_KEYS + kind * 64
        logits[:, start : start + R] = per_region[:, :, kind]
    mean = h.sum(1) / count
    value = net.val2(F.relu(net.val1(torch.cat([g, mean], -1))))
    return logits, value, joined(net.own, [h], g), h, g, mask


def pooled(x, mask):
    """Moyenne et maximum de `x` (B, …, R, h) sur les régions de `mask` (B, …, R) ; zéros si aucune."""
    m = mask[..., None]
    count = m.sum(-2).clamp(min=1)
    mean = (x * m).sum(-2) / count
    mx = (x + (1 - m) * -1e9).max(-2).values * (m.sum(-2) > 0)
    return torch.cat([mean, mx], -1)


class Heads(nn.Module):
    def __init__(self, base, plain=False):
        super().__init__()
        self.base = base
        self.plain = plain
        h = base.hidden
        self.card = nn.Sequential(nn.Linear(M + SLOTS + h, h), nn.ReLU(), nn.Linear(h, h))
        self.pair_region = nn.Linear(h, h)
        self.pair_card = nn.Linear(h, h, bias=False)
        self.card_out = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, 1))
        self.dec_region = nn.Linear(2 * h, h)
        self.dec_out = nn.Sequential(nn.Linear(5 * h, h), nn.ReLU(), nn.Linear(h, 1))
        for lin in (self.card_out[-1], self.dec_out[-1]):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.register_buffer("slots", torch.eye(SLOTS), persistent=False)

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("base.")]

    def forward(self, node, glob, adj, regions):
        logits, value, own, h, g, mask = trunk(self.base, node, glob, adj, regions)
        if self.plain:
            return logits, value, own
        B = node.shape[0]
        # peuple : les cartes du marché, une même tête pour les six places
        scale = self.base.global_scale[G_MARKET : G_MARKET + SLOTS * M].view(SLOTS, M)
        cards = glob[:, G_MARKET : G_MARKET + SLOTS * M].float().view(B, SLOTS, M) * scale
        z = self.card(torch.cat([cards, self.slots.expand(B, -1, -1), g[:, None].expand(-1, SLOTS, -1)], -1))
        pair = F.relu(self.pair_region(h)[:, None] + self.pair_card(z)[:, :, None])  # (B, 6, R, h)
        pick = self.card_out(torch.cat([pooled(pair, mask[:, None].expand(-1, SLOTS, -1)), z], -1))[..., 0]
        # déclin : chaque région, regroupée sur les miennes (peuple actif) et sur toutes
        u = F.relu(self.dec_region(torch.cat([h, g[:, None].expand_as(h)], -1)))
        mine = mask * (node[..., N_OWNER] == 1).float() * (node[..., N_ACTIVE] == 1).float()
        decline = self.dec_out(torch.cat([pooled(u, mine), pooled(u, mask), g], -1))[:, 0]
        extra = torch.zeros_like(logits)
        extra[:, :SLOTS] = pick
        extra[:, DECLINE] = decline
        return logits + extra, value, own
