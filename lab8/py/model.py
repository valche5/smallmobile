"""Le réseau de lab8, côté PyTorch : encodeur transformer, têtes de valeur, décodeur de plans.
Même calcul que lab8/brain/src/net.rs, qui le refait en Rust (comparés par check.py).

Entrées d'un lot (voir data.py) : node (B, 48, 77) u8, glob (B, 820) u8, dist (B, 48, 48) u8,
regions (B,), prev (B, S) : coup précédent de chaque pas (START au premier), dyn (B, S, 20) u8.

    python model.py ../nets/gen0.bin [--d 256 --layers 6 --heads 8 --dec 2]   réseau neuf
"""

import argparse
import struct

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REGIONS, NODE, GLOBAL = 48, 77, 820
PLAYERS, MARKET = 5, 6
TOKENS = REGIONS + PLAYERS + MARKET + 1
T_PLAYER, T_MARKET, T_GAME = REGIONS, REGIONS + PLAYERS, TOKENS - 1
KEYS, GLOBAL_KEYS, KINDS = 608, 32, 9
RELATIONS, FAR = 10, 4
DYN = 20
COUNTS = 16
OWNERS = 6
START = 65535
NEG = -1e9
# la tête de valeur prédit les pièces gagnées divisées par ce nombre
GAIN_SCALE = 50.0

# les blocs de la partie dans l'encodage de lab7 (lab7/brain/src/encode.rs) ; tokens.rs
G_PLAYER, P = 67, 81
G_MARKET, M = G_PLAYER + PLAYERS * P, 38
G_HABITS, HABITS = G_MARKET + MARKET * M, 14
G_ATTACKS = G_HABITS + PLAYERS * HABITS
G_RECENT = G_ATTACKS + PLAYERS * PLAYERS
DYN_GLOBAL = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 63]
N_OWNER, N_ACTIVE, N_DECLINED = 17, 22, 23


def player_index(r):
    """Les entrées de la partie du jeton du joueur relatif r (tokens::player)."""
    idx = [G_PLAYER + r * P + i for i in range(P)]
    idx += [G_HABITS + r * HABITS + i for i in range(HABITS)]
    idx += [G_ATTACKS + r * PLAYERS + t for t in range(PLAYERS)]
    idx += [G_ATTACKS + t * PLAYERS + r for t in range(PLAYERS)]
    idx += [G_RECENT + r * PLAYERS + t for t in range(PLAYERS)]
    idx += [G_RECENT + t * PLAYERS + r for t in range(PLAYERS)]
    return idx


PLAYER_INDEX = np.array([player_index(r) for r in range(PLAYERS)])  # (5, 115)
MARKET_INDEX = np.array([[G_MARKET + m * M + i for i in range(M)] for m in range(MARKET)])  # (6, 38)
PLAYER_IN, MARKET_IN, GAME_IN = PLAYER_INDEX.shape[1], M, G_PLAYER


def lab7_scales():
    """Les échelles des entrées de lab7 (lab7/py/model.py, `scales`)."""
    node = np.ones(NODE, np.float32)
    node[13] = 0.1
    node[14] = 0.1
    for i in (58, 65, 68, 69, 73):
        node[i] = 0.2
    node[59] = 0.5
    node[76] = 0.25
    glob = np.ones(GLOBAL, np.float32)
    glob[4:7] = 0.1
    glob[18:20] = 0.2
    glob[22] = 0.33
    glob[26] = 0.25
    glob[28] = 0.1
    glob[63] = 0.5
    glob[64:67] = 0.1
    for q in range(PLAYERS):
        b = G_PLAYER + q * P
        glob[b + 1] = 0.02
        glob[b + 2] = 0.1
        glob[b + 3] = 0.1
        glob[b + 38] = 0.1
        glob[b + 39] = 0.1
        glob[b + 40] = 0.05
        glob[b + 75] = 0.1
    for m in range(MARKET):
        b = G_MARKET + m * M
        glob[b + 35] = 0.2
        glob[b + 36] = 0.1
    glob[G_HABITS:G_ATTACKS] = 0.1
    for q in range(PLAYERS):
        glob[G_HABITS + q * HABITS + 13] = 0.05
    glob[G_ATTACKS:G_RECENT] = 0.1
    glob[G_RECENT : G_RECENT + 25] = 0.25
    return node, glob


def scales():
    """Échelles de chaque sorte de jeton, et des entrées d'un pas : exportées avec les poids."""
    node, glob = lab7_scales()
    return {
        "region_scale": node,
        "player_scale": glob[PLAYER_INDEX[0]],
        "market_scale": glob[MARKET_INDEX[0]],
        "game_scale": glob[:GAME_IN],
        "dyn_scale": np.concatenate([glob[DYN_GLOBAL], np.ones(2, np.float32)]),
    }


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x, mem, bias):
        """x (B, Tq, d) regarde mem (B, Tk, d) ; bias (B ou 1, H ou 1, Tq, Tk) ajouté aux scores."""
        B, Tq, d = x.shape
        Tk, h = mem.shape[1], self.heads
        q = self.q(x).view(B, Tq, h, d // h).transpose(1, 2)
        k = self.k(mem).view(B, Tk, h, d // h).transpose(1, 2)
        v = self.v(mem).view(B, Tk, h, d // h).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.o(out.transpose(1, 2).reshape(B, Tq, d))


class Mlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a, self.b = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return self.b(F.gelu(self.a(x), approximate="tanh"))


class Layer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.mlp = Attn(d, heads), Mlp(d)
        # biais de l'attention par relation et par tête
        self.rel = nn.Embedding(RELATIONS, heads)
        nn.init.zeros_(self.rel.weight)


class DecLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.me, self.cross, self.mlp = Attn(d, heads), Attn(d, heads), Mlp(d)


class Net(nn.Module):
    def __init__(self, d=256, layers=6, heads=8, dec=2):
        super().__init__()
        self.d = d
        self.region_in = nn.Linear(NODE, d)
        self.player_in = nn.Linear(PLAYER_IN, d)
        self.market_in = nn.Linear(MARKET_IN, d)
        self.game_in = nn.Linear(GAME_IN, d)
        self.player_pos = nn.Embedding(PLAYERS, d)
        self.market_pos = nn.Embedding(MARKET, d)
        self.layers = nn.ModuleList(Layer(d, heads) for _ in range(layers))
        self.enc_ln = nn.LayerNorm(d)
        self.val1, self.val2 = nn.Linear(2 * d, d), nn.Linear(d, 2 * PLAYERS)
        self.next1, self.next2 = nn.Linear(2 * d, d), nn.Linear(d, 2 * PLAYERS)
        # à l'apprentissage seulement : à qui sera chaque région au prochain tour du joueur
        self.own = nn.Linear(d, OWNERS)
        self.dec_start = nn.Parameter(torch.zeros(d))
        self.dec_glob = nn.Embedding(GLOBAL_KEYS, d)
        self.dec_kind = nn.Embedding(KINDS, d)
        self.dec_reg = nn.Linear(d, d)
        self.dec_dyn = nn.Linear(DYN, d)
        self.dec_count = nn.Embedding(COUNTS, d)
        self.dec_layers = nn.ModuleList(DecLayer(d, heads) for _ in range(dec))
        self.dec_ln = nn.LayerNorm(d)
        self.out_glob = nn.Linear(d, GLOBAL_KEYS)
        self.out_kind = nn.Linear(d, KINDS * d)
        # sorties à zéro : un réseau neuf joue au hasard, mêmes chances pour tous
        for lin in (self.val2, self.next2, self.own, self.out_glob, self.out_kind):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        for name, v in scales().items():
            self.register_buffer(name, torch.from_numpy(v), persistent=False)
        self.register_buffer("player_index", torch.from_numpy(PLAYER_INDEX), persistent=False)
        self.register_buffer("market_index", torch.from_numpy(MARKET_INDEX), persistent=False)

    def encode(self, node, glob, dist, regions):
        """Sorties de l'encodeur (B, TOKENS, d), jetons présents (B, TOKENS), têtes (B, 20), propriétaires
        (B, 48, OWNERS)."""
        B = node.shape[0]
        dev = node.device
        node_f = node.float() * self.region_scale
        glob_f = glob.float()
        players = glob_f[:, self.player_index] * self.player_scale  # (B, 5, 115)
        market = glob_f[:, self.market_index] * self.market_scale  # (B, 6, 38)
        game = glob_f[:, :GAME_IN] * self.game_scale
        x = torch.cat(
            [
                self.region_in(node_f),
                self.player_in(players) + self.player_pos.weight,
                self.market_in(market) + self.market_pos.weight,
                self.game_in(game)[:, None],
            ],
            1,
        )
        region_ok = torch.arange(REGIONS, device=dev)[None] < regions[:, None]
        player_ok = glob[:, G_PLAYER + P * torch.arange(PLAYERS, device=dev)] != 0
        market_ok = glob[:, G_MARKET + M * torch.arange(MARKET, device=dev)] != 0
        valid = torch.cat([region_ok, player_ok, market_ok, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
        x = x * valid[..., None]
        rel = relations(node, dist, region_ok)
        key_mask = torch.where(valid, 0.0, NEG)[:, None, None, :]  # (B, 1, 1, T)
        # biais des relations de toutes les couches d'un coup, par un produit avec le one-hot des
        # relations plutôt que `layer.rel(rel)` : le gradient d'une table de 10 lignes lue 921 600 fois
        # par lot prenait 62 % du temps de l'apprentissage. Un produit par position (`bmm`) : le
        # gradient de la table en est un aussi, au lieu d'un seul produit de 10 × 48 sur 921 600
        # termes, très lent. En f32 : les mêmes valeurs qu'une lecture.
        with torch.autocast(dev.type, enabled=False):
            table = torch.cat([layer.rel.weight for layer in self.layers], 1)  # (R, L·H)
            onehot = F.one_hot(rel, RELATIONS).float().view(B, -1, RELATIONS)  # (B, T·T, R)
            biases = torch.bmm(onehot, table.expand(B, *table.shape))  # (B, T·T, L·H)
        biases = biases.view(*rel.shape, len(self.layers), -1).permute(3, 0, 4, 1, 2)  # (L, B, H, T, T)
        for layer, rel_bias in zip(self.layers, biases):
            bias = rel_bias + key_mask  # (B, H, T, T)
            a = layer.ln1(x)
            x = x + layer.attn(a, a, bias)
            x = x + layer.mlp(layer.ln2(x))
        out = self.enc_ln(x)
        rmask = region_ok.float()
        mean = (out[:, :REGIONS] * rmask[..., None]).sum(1) / rmask.sum(1, keepdim=True)
        pooled = torch.cat([out[:, T_GAME], mean], -1)
        heads = torch.cat([self.val2(F.relu(self.val1(pooled))), self.next2(F.relu(self.next1(pooled)))], -1)
        return out, valid, heads, self.own(out[:, :REGIONS])

    def decode(self, out, valid, regions, prev, dyn):
        """Logits des coups (B, S, KEYS) de chaque pas, en suivant les coups `prev` (forçage par
        l'enseignant ; tous les pas en parallèle, attention causale)."""
        B, S = prev.shape
        d = self.d
        dev = out.device
        is_region = (prev != START) & (prev >= GLOBAL_KEYS)
        is_glob = (prev != START) & (prev < GLOBAL_KEYS)
        rk = torch.where(is_region, prev - GLOBAL_KEYS, 0)
        kind, region = rk // 64, rk % 64
        reg = self.dec_reg(out[:, :REGIONS])  # (B, 48, d)
        reg_in = torch.gather(reg, 1, region.clamp(max=REGIONS - 1)[..., None].expand(B, S, d))
        e = torch.where(is_region[..., None], self.dec_kind(kind) + reg_in, 0.0)
        e = e + torch.where(is_glob[..., None], self.dec_glob(torch.where(is_glob, prev, 0)), 0.0)
        e = e + torch.where((prev == START)[..., None], self.dec_start, 0.0)
        # coups déjà joués sur la région du coup précédent, celui-ci compris
        onehot = F.one_hot(region, 64) * is_region[..., None]
        counts = torch.gather(onehot.cumsum(1), 2, region[..., None])[..., 0] * is_region
        e = e + self.dec_count(counts.clamp(max=COUNTS - 1))
        x = e + self.dec_dyn(dyn.float() * self.dyn_scale)
        causal = torch.triu(torch.full((S, S), NEG, device=dev), 1)[None, None]
        cross_mask = torch.where(valid, 0.0, NEG)[:, None, None, :]
        for layer in self.dec_layers:
            a = layer.ln1(x)
            x = x + layer.me(a, a, causal)
            x = x + layer.cross(layer.ln2(x), out, cross_mask)
            x = x + layer.mlp(layer.ln3(x))
        n = self.dec_ln(x)
        q = self.out_kind(n).view(B, S, KINDS, d)
        per_region = torch.einsum("bskd,brd->bskr", q, out[:, :REGIONS]) / d**0.5  # (B, S, 9, 48)
        region_ok = torch.arange(REGIONS, device=dev)[None] < regions[:, None]
        per_region = per_region.masked_fill(~region_ok[:, None, None, :], NEG)
        pad = torch.full((B, S, KINDS, 64 - REGIONS), NEG, device=dev, dtype=per_region.dtype)
        return torch.cat([self.out_glob(n), torch.cat([per_region, pad], -1).reshape(B, S, KINDS * 64)], -1)

    def forward(self, node, glob, dist, regions, prev, dyn):
        out, valid, heads, owners = self.encode(node, glob, dist, regions)
        return heads, owners, self.decode(out, valid, regions, prev, dyn)


def relations(node, dist, region_ok):
    """Relations entre jetons (B, T, T) : tokens::relations."""
    B = node.shape[0]
    dev = node.device
    rel = torch.zeros(B, TOKENS, TOKENS, dtype=torch.long, device=dev)
    pair = region_ok[:, :, None] & region_ok[:, None, :]
    rel[:, :REGIONS, :REGIONS] = torch.where(pair, 1 + dist.long().clamp(max=FAR), 0)
    owner = node[:, :, N_OWNER : N_OWNER + PLAYERS] != 0  # (B, 48, 5)
    declined = (node[:, :, N_DECLINED] != 0)[..., None]
    to = torch.where(owner, torch.where(declined, 7, 6), 0) * region_ok[..., None]
    rel[:, :REGIONS, T_PLAYER : T_PLAYER + PLAYERS] = to
    rel[:, T_PLAYER : T_PLAYER + PLAYERS, :REGIONS] = torch.where(to > 0, to + 2, 0).transpose(1, 2)
    return rel


def load(path) -> Net:
    """Un réseau écrit par `export` ; tailles lues dans le fichier."""
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
    d = tensors["region_in.weight"].shape[0]
    heads = tensors["layers.0.rel.weight"].shape[1]
    layers = sum(1 for k in tensors if k.startswith("layers.") and k.endswith(".ln1.weight"))
    dec = sum(1 for k in tensors if k.startswith("dec_layers.") and k.endswith(".ln1.weight"))
    net = Net(d, layers, heads, dec)
    for name, v in scales().items():
        assert np.array_equal(tensors.pop(name).numpy(), v), f"{path} : échelles différentes ({name})"
    net.load_state_dict(tensors)
    return net


def export(net: Net, path):
    """Écrit les poids au format SMNN1 lu par lab8/brain/src/net.rs, avec les échelles."""
    tensors = [(k, v.detach().float().cpu().contiguous()) for k, v in net.state_dict().items()]
    tensors += [(name, torch.from_numpy(v)) for name, v in scales().items()]
    with open(path, "wb") as f:
        f.write(b"SMNN1")
        f.write(struct.pack("<I", len(tensors)))
        for name, t in tensors:
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", t.dim()))
            for s in t.shape:
                f.write(struct.pack("<I", s))
            f.write(t.numpy().astype("<f4").tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dec", type=int, default=2)
    a = ap.parse_args()
    torch.manual_seed(0)
    net = Net(a.d, a.layers, a.heads, a.dec)
    export(net, a.out)
    print(f"{sum(p.numel() for p in net.parameters()):,} paramètres")
