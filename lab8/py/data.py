"""Les parties d'imitation écrites par `lab8 record` (format : lab8/src/record.rs), en lots pour
l'apprentissage.

Chaque fil a ses fichiers (`r{graine}-{fil}.seg`, `.node`…), mappés tels quels. Un lot rassemble des
plans : l'entrée de l'encodeur au début de chacun, et tous ses pas pour le décodeur (forçage par
l'enseignant)."""

from pathlib import Path

import numpy as np
import torch

from model import DYN, FAR, GLOBAL, KEYS, NODE, REGIONS, START

SEG = np.dtype(
    [
        ("players", "u1"),
        ("regions", "u1"),
        ("steps", "<u2"),
        ("phase", "u1"),
        ("pad", "u1", 3),
        ("win", "<f4", 5),
        ("gain", "<f4", 5),
        ("here", "<f4", 5),
        ("next_win", "<f4", 5),
        ("next_gain", "<f4", 5),
        ("prior_win", "<f4", 5),
    ]
)
STEP = np.dtype([("key", "<u2"), ("count", "<u2"), ("dyn", "u1", DYN), ("pad", "u1", 2)])
LEGAL = np.dtype([("key", "<u2"), ("prob", "<u2"), ("prior", "<u2")])
PROBABILITY = 60000.0
OFF_MAP = 255
# plus long plan gardé en entier (au-delà, ses derniers pas sont coupés : rare)
MAX_STEPS = 64
# part des parties gardées pour la validation : celles dont la donne est un multiple de VAL_EVERY
VAL_EVERY = 20


def offsets(counts):
    out = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(counts, out=out[1:])
    return out


class File:
    """Les plans d'un fil."""

    def __init__(self, prefix):
        m = lambda ext, dt: np.memmap(f"{prefix}{ext}", dt, "r") if Path(f"{prefix}{ext}").stat().st_size else np.zeros(0, dt)
        self.seg = np.fromfile(f"{prefix}.seg", SEG)
        self.node = m(".node", np.uint8).reshape(-1, NODE)
        self.glob = m(".glob", np.uint8).reshape(-1, GLOBAL)
        self.adj = m(".adj", "<u8")
        self.owner = m(".owner", np.uint8)
        self.step = m(".step", STEP)
        self.legal = m(".legal", LEGAL)
        self.r_off = offsets(self.seg["regions"])
        self.s_off = offsets(self.seg["steps"])
        self.l_off = offsets(self.step["count"])
        # les plans de chaque partie, et sa donne
        games = [line.split() for line in Path(f"{prefix}.games").read_text().splitlines()]
        per_game = np.array([int(g[4]) for g in games], np.int64)
        deal = np.array([int(g[0]) for g in games], np.uint64)
        self.val = np.repeat(deal % VAL_EVERY == 0, per_game)
        self.game = np.repeat(np.arange(len(games)), per_game)
        assert len(self.val) == len(self.seg), f"{prefix} : .games ne suit pas .seg"

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
        node = self.node[rows] * ok[..., None]
        adj = np.where(ok, self.adj[rows], 0)
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
        within = np.arange(len(flat)) - np.repeat(offsets(counts)[:-1], counts)
        lg = self.legal[np.repeat(self.l_off[srows].ravel(), counts) + within]
        return {
            "players": s["players"].astype(np.int64),
            "regions": regions,
            "phase": s["phase"].astype(np.int64),
            "node": node,
            "glob": np.asarray(self.glob[idx]),
            "adj": adj.astype(np.int64),
            "owner": owner.astype(np.int64),
            "win": s["win"],
            "gain": s["gain"],
            "here": s["here"],
            "next_win": s["next_win"],
            "next_gain": s["next_gain"],
            "prior_win": s["prior_win"],
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
