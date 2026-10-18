"""Les parties d'entraînement écrites par `lab selfplay` (voir lab/src/selfplay.rs pour le format)."""

from pathlib import Path

import numpy as np

KEYS = 448
REGIONS = 48
NODE = 77
GLOBAL = 700
ILLEGAL = 65535

FIELDS = (
    [
        ("players", "u1"),
        ("regions", "u1"),
        ("full", "u1"),
        ("spare", "u1"),
        ("win", "<f4", 5),
        ("gain", "<f4", 5),
        ("visits", "<u2", KEYS),
        ("adj", "<u8", REGIONS),
        ("node", "u1", (REGIONS, NODE)),
        ("glob", "u1", GLOBAL),
    ]
)
# anciens fichiers (*.bin) : sans les pièces attendues par la recherche
RECORD_V1 = np.dtype(FIELDS)
# fichiers *.v2.bin : pièces encore attendues par la recherche pour chaque joueur relatif (NaN : inconnu)
RECORD = np.dtype(FIELDS + [("qgain", "<f4", 5)])


# les positions d'une même partie se suivent dans un fichier : la validation se fait par blocs
BLOCK = 256


class Records:
    """Les enregistrements lus à la demande dans les fichiers (`np.memmap`) : pour des données plus
    grosses que la mémoire. `recs[sel]` (indices croissants) renvoie un tableau `RECORD`."""

    def __init__(self, sources):
        # sources : (fichier, dtype, nombre d'enregistrements du fichier, lignes gardées)
        self.maps = [
            np.memmap(f, dtype=dtype, mode="r", shape=(count,)) if count else np.zeros(0, dtype=dtype)
            for f, dtype, count, _ in sources
        ]
        self.starts = np.cumsum([0] + [len(rows) for *_, rows in sources])
        self.rows = np.concatenate([rows for *_, rows in sources]) if sources else np.zeros(0, np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, sel):
        sel = np.asarray(sel)
        out = np.empty(len(sel), dtype=RECORD)
        # les indices sont croissants : chaque fichier en occupe une tranche
        cuts = np.searchsorted(sel, self.starts)
        for i, m in enumerate(self.maps):
            a, b = cuts[i], cuts[i + 1]
            if a == b:
                continue
            fill(out[a:b], m[self.rows[sel[a:b]]])
        return out


def fill(dst, src):
    """Copie des enregistrements (anciens ou nouveaux) dans un tableau `RECORD`."""
    if src.dtype == RECORD:
        dst[...] = src
    else:
        for name in RECORD_V1.names:
            dst[name] = src[name]
        dst["qgain"] = np.nan


def load(paths, seed=0, with_value=False, lazy=False):
    """Les enregistrements des fichiers (ou dossiers) donnés. `dossier@0.2` n'en garde que 20 %, par
    blocs de positions consécutives (des parties entières, à peu près) ; `dossier:policy` ne sert
    qu'à la politique (sa valeur n'est pas apprise).

    Renvoie les enregistrements, le numéro de bloc de chacun (pour séparer la validation) et, si
    `with_value`, si sa valeur s'apprend. `lazy` : les enregistrements restent dans les fichiers
    (`Records`), lus à chaque lot."""
    rng = np.random.default_rng(seed)
    # premier passage : quels enregistrements garder (sans rien lire), pour tout écrire d'un coup
    # dans un seul tableau (pas de copie : les données peuvent occuper la moitié de la mémoire)
    plan = []
    next_block = 0
    for spec in paths:
        spec, _, tag = str(spec).partition(":")
        path, _, share = spec.partition("@")
        share = float(share) if share else 1.0
        p = Path(path)
        for f in sorted(p.rglob("*.bin")) if p.is_dir() else [p]:
            dtype = RECORD if f.name.endswith(".v2.bin") else RECORD_V1
            count = f.stat().st_size // dtype.itemsize
            ids = np.arange(count) // BLOCK + next_block
            next_block = int(ids[-1]) + 1 if count else next_block
            mask = None
            if share < 1.0:
                keep_blocks = rng.random(next_block) < share
                mask = keep_blocks[ids]
                ids = ids[mask]
            plan.append((f, dtype, count, mask, ids, tag != "policy"))
    total = sum(len(ids) for *_, ids, _ in plan)
    blocks = np.concatenate([ids for *_, ids, _ in plan]) if plan else np.zeros(0, np.int64)
    values = np.concatenate([np.full(len(ids), v) for *_, ids, v in plan]) if plan else np.zeros(0, bool)
    if lazy:
        sources = [
            (f, dtype, count, np.arange(count) if mask is None else np.flatnonzero(mask))
            for f, dtype, count, mask, _, _ in plan
        ]
        recs = Records(sources)
        return (recs, blocks, values) if with_value else (recs, blocks)
    recs = np.empty(total, dtype=RECORD)
    at = 0
    for f, dtype, count, mask, ids, _ in plan:
        src = np.memmap(f, dtype=dtype, mode="r", shape=(count,)) if count else np.zeros(0, dtype=dtype)
        if mask is not None:
            src = src[mask]
        fill(recs[at : at + len(ids)], src)
        at += len(ids)
        del src
    return (recs, blocks, values) if with_value else (recs, blocks)
