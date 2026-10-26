"""Les parties d'imitation écrites par `lab8 record` (forme compacte : lab8/src/record.rs), en lots pour
l'apprentissage.

Chaque fil a ses fichiers (`r{graine}-{fil}.seg`, `.nmask`…), lus entiers en mémoire. Un lot rassemble
des plans : l'entrée de l'encodeur au début de chacun, redonnée en entier, et tous ses pas pour le
décodeur (forçage par l'enseignant).

    python data.py DIR      vérifie la forme compacte face aux entrées pleines (`lab8 record --raw true`)"""

from pathlib import Path

import numpy as np
import torch

from model import DYN, FAR, GLOBAL, KEYS, NODE, REGIONS, START

SEG = np.dtype(
    [
        ("game", "<u4"),
        ("steps", "<u2"),
        ("node_nz", "<u2"),
        ("glob_nz", "<u2"),
        ("players", "u1"),
        ("regions", "u1"),
        ("phase", "u1"),
        ("pad", "u1", 3),
        # win, gain, here, next_win, next_gain, prior_win : 5 chacune
        ("values", "<f2", 30),
    ]
)
VALUES = ("win", "gain", "here", "next_win", "next_gain", "prior_win")
STEP = np.dtype([("key", "<u2"), ("count", "<u2"), ("dyn", "u1", DYN)])
LEGAL = np.dtype([("key", "<u2"), ("prob", "<u2"), ("prior", "<u2")])
PROBABILITY = 60000.0
# colonnes des régions données par la carte, une fois par partie (record.rs, MAP_COLS)
MAP_COLS = 15
DYN_COLS = NODE - MAP_COLS
OFF_MAP = 255
# plus long plan gardé en entier (au-delà, ses derniers pas sont coupés : rare)
MAX_STEPS = 64
# part des parties gardées pour la validation : celles dont la donne est un multiple de VAL_EVERY
VAL_EVERY = 20


def offsets(counts):
    out = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(counts, out=out[1:])
    return out


def spans(starts, counts):
    """Les indices des tranches [start, start + count) mises bout à bout."""
    counts = np.asarray(counts, np.int64)
    within = np.arange(counts.sum()) - np.repeat(offsets(counts)[:-1], counts)
    return np.repeat(np.asarray(starts, np.int64), counts) + within


class File:
    """Les plans d'un fil, en mémoire (forme compacte de lab8/src/record.rs)."""

    def __init__(self, prefix):
        r = lambda ext, dt: np.fromfile(f"{prefix}{ext}", dt)
        self.seg = r(".seg", SEG)
        self.values = self.seg["values"].astype(np.float32)
        self.map = r(".map", np.uint8).reshape(-1, MAP_COLS)
        self.adj = r(".adj", "<u8").astype(np.int64)
        self.nmask = r(".nmask", "<u8")
        self.nval = r(".nval", np.uint8)
        self.gmask = r(".gmask", np.uint8).reshape(-1, (GLOBAL + 7) // 8)
        self.gval = r(".gval", np.uint8)
        self.owner = r(".owner", np.uint8)
        self.step = r(".step", STEP)
        self.legal = r(".legal", LEGAL)
        regions = self.seg["regions"].astype(np.int64)
        self.r_off = offsets(regions)
        self.nv_off = offsets(self.seg["node_nz"])
        self.gv_off = offsets(self.seg["glob_nz"])
        self.s_off = offsets(self.seg["steps"])
        self.l_off = offsets(self.step["count"])
        # les parties : leurs plans, leur donne, leur carte
        games = [line.split() for line in Path(f"{prefix}.games").read_text().splitlines()]
        per_game = np.array([int(g[4]) for g in games], np.int64)
        deal = np.array([int(g[0]) for g in games], np.uint64)
        assert per_game.sum() == len(self.seg), f"{prefix} : .games ne suit pas .seg"
        first = offsets(per_game)[:-1][per_game > 0]
        game_regions = np.zeros(len(games), np.int64)
        game_regions[per_game > 0] = regions[first]
        self.m_off = offsets(game_regions)
        self.val = np.repeat(deal % VAL_EVERY == 0, per_game)
        assert len(self.map) == self.m_off[-1] and len(self.adj) == self.m_off[-1]

    def __len__(self):
        return len(self.seg)

    def take(self, idx):
        """Les plans `idx` (tableaux numpy, régions remplies jusqu'à REGIONS, pas jusqu'au plus long)."""
        s = self.seg[idx]
        B = len(idx)
        regions = s["regions"].astype(np.int64)
        ar = np.arange(REGIONS)
        ok = ar[None] < regions[:, None]
        rows = np.where(ok, self.r_off[idx][:, None] + ar[None], 0)
        maps = np.where(ok, self.m_off[s["game"].astype(np.int64)][:, None] + ar[None], 0)
        # les régions : colonnes de la carte, puis les non nuls des autres
        node = np.zeros((B, REGIONS, NODE), np.uint8)
        node[:, :, :MAP_COLS] = self.map[maps] * ok[..., None]
        bits = (self.nmask[rows][..., None] >> np.arange(DYN_COLS, dtype=np.uint64)) & np.uint64(1)
        bits = bits.astype(bool) & ok[..., None]
        dyn_cols = np.zeros((B, REGIONS, DYN_COLS), np.uint8)
        dyn_cols[bits] = self.nval[spans(self.nv_off[idx], s["node_nz"])]
        node[:, :, MAP_COLS:] = dyn_cols
        glob = np.zeros((B, GLOBAL), np.uint8)
        gbits = np.unpackbits(self.gmask[idx], axis=1, bitorder="little")[:, :GLOBAL].astype(bool)
        glob[gbits] = self.gval[spans(self.gv_off[idx], s["glob_nz"])]
        adj = np.where(ok, self.adj[maps], 0)
        owner = np.where(ok, self.owner[rows], OFF_MAP)
        steps = np.minimum(s["steps"].astype(np.int64), MAX_STEPS)
        S = int(steps.max())
        ar = np.arange(S)
        sok = ar[None] < steps[:, None]
        srows = np.where(sok, self.s_off[idx][:, None] + ar[None], 0)
        st = self.step[srows]
        keys = np.where(sok, st["key"], 0).astype(np.int64)
        prev = np.full((B, S), START, np.int64)
        prev[:, 1:] = keys[:, :-1]
        dyn = st["dyn"] * sok[..., None]
        # les coups possibles de chaque pas, à plat ; `flat` : leur pas (plan × S + pas)
        counts = np.where(sok, st["count"], 0).astype(np.int64).ravel()
        flat = np.repeat(np.arange(B * S), counts)
        lg = self.legal[spans(self.l_off[srows].ravel(), counts)]
        v = self.values[idx]
        out = {name: v[:, 5 * i : 5 * i + 5] for i, name in enumerate(VALUES)}
        return out | {
            "players": s["players"].astype(np.int64),
            "regions": regions,
            "phase": s["phase"].astype(np.int64),
            "node": node,
            "glob": glob,
            "adj": adj,
            "owner": owner.astype(np.int64),
            "steps": steps,
            "key": keys,
            "prev": prev,
            "dyn": dyn,
            "flat": flat,
            "legal_key": lg["key"].astype(np.int64),
            "legal_prob": lg["prob"].astype(np.float32) / PROBABILITY,
            "legal_prior": lg["prior"].astype(np.float32) / PROBABILITY,
        }


class Data:
    """Les plans de tous les fils de `dirs`, indexés à plat (fichier, plan)."""

    def __init__(self, dirs):
        prefixes = sorted(str(p)[: -len(".seg")] for d in dirs for p in Path(d).glob("*.seg"))
        assert prefixes, f"aucun fichier .seg dans {dirs}"
        self.files = [File(p) for p in prefixes]
        self.file = np.concatenate([np.full(len(f), i, np.int32) for i, f in enumerate(self.files)])
        self.local = np.concatenate([np.arange(len(f), dtype=np.int64) for f in self.files])
        self.val = np.concatenate([f.val for f in self.files])
        self.phase = np.concatenate([f.seg["phase"] for f in self.files])
        self.players = np.concatenate([f.seg["players"] for f in self.files])

    def __len__(self):
        return len(self.file)

    def take(self, idx):
        """Le lot des plans `idx` (indices à plat), tableaux numpy."""
        idx = np.sort(idx)
        parts = []
        for f in np.unique(self.file[idx]):
            parts.append(self.files[f].take(self.local[idx[self.file[idx] == f]]))
        if len(parts) == 1:
            return parts[0]
        S = max(p["key"].shape[1] for p in parts)
        out, base = {}, 0
        for p in parts:
            s, B = p["key"].shape[1], len(p["players"])
            if s < S:
                for name, fill in (("key", 0), ("prev", START), ("dyn", 0)):
                    pad = np.full((B, S - s) + p[name].shape[2:], fill, p[name].dtype)
                    p[name] = np.concatenate([p[name], pad], 1)
            # les coups possibles, rangés par (plan, pas) dans le lot entier
            b, t = np.divmod(p["flat"], s)
            p["flat"] = (b + base) * S + t
            base += B
        for name in parts[0]:
            out[name] = np.concatenate([p[name] for p in parts])
        return out


@torch.autocast("cuda", enabled=False)
def distances(adj, regions):
    """Distances en voisinages entre les régions (B, 48, 48), plafonnées à FAR : tokens::distances.
    En f32, sur des tenseurs contigus : le produit par lots bf16 de ROCm 7.14 lit hors de la mémoire
    sur un tenseur étiré (pas nul entre les lots)."""
    B = adj.shape[0]
    dev = adj.device
    bits = ((adj[..., None] >> torch.arange(REGIONS, device=dev)) & 1).float()  # (B, 48, 48)
    eye = torch.eye(REGIONS, device=dev, dtype=torch.bool).expand(B, -1, -1).contiguous()
    dist = torch.full((B, REGIONS, REGIONS), FAR, dtype=torch.uint8, device=dev)
    dist[eye] = 0
    seen, front = eye.clone(), eye.float()
    for step in range(1, FAR):
        nxt = (torch.bmm(front, bits) > 0) & ~seen
        dist[nxt] = step
        seen |= nxt
        front = nxt.float()
    return dist


def to_device(b, device):
    """Le lot sur le GPU, avec les cibles des coups en tenseurs pleins (B, S, KEYS)."""
    t = {name: torch.from_numpy(np.ascontiguousarray(v)).to(device) for name, v in b.items()}
    B, S = t["key"].shape
    legal = torch.zeros(B * S, KEYS, dtype=torch.bool, device=device)
    legal[t["flat"], t["legal_key"]] = True
    policy = torch.zeros(B * S, KEYS, device=device)
    policy[t["flat"], t["legal_key"]] = t["legal_prob"]
    prior = torch.zeros(B * S, KEYS, device=device)
    prior[t["flat"], t["legal_key"]] = t["legal_prior"]
    t["legal"], t["policy"], t["prior"] = legal.view(B, S, KEYS), policy.view(B, S, KEYS), prior.view(B, S, KEYS)
    t["step_ok"] = torch.arange(S, device=device)[None] < t["steps"][:, None]
    t["dist"] = distances(t["adj"], t["regions"])
    return t


def check(directory):
    """La forme compacte redonne-t-elle les entrées pleines (`.raw`) ?"""
    from model import NODE as N
    plans = 0
    for p in sorted(Path(directory).glob("*.seg")):
        prefix = str(p)[: -len(".seg")]
        f = File(prefix)
        raw = np.fromfile(f"{prefix}.raw", np.uint8)
        b = f.take(np.arange(len(f)))
        at = 0
        for i in range(len(f)):
            r = int(b["regions"][i])
            node = raw[at : at + r * N].reshape(r, N)
            glob = raw[at + r * N : at + r * N + GLOBAL]
            values = raw[at + r * N + GLOBAL : at + r * N + GLOBAL + 120].view("<f4")
            at += r * N + GLOBAL + 120
            assert np.array_equal(b["node"][i, :r], node) and not b["node"][i, r:].any(), f"{prefix} : régions du plan {i}"
            assert np.array_equal(b["glob"][i], glob), f"{prefix} : partie du plan {i}"
            got = np.concatenate([b[name][i] for name in VALUES])
            assert np.allclose(got, values, rtol=1e-3, atol=1e-4), f"{prefix} : valeurs du plan {i}"
        assert at == len(raw)
        plans += len(f)
    print(f"{plans} plans : forme compacte identique aux entrées pleines")


if __name__ == "__main__":
    import sys

    check(sys.argv[1])
