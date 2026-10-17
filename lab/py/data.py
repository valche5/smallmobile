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


def load(paths, seed=0):
    """Les enregistrements des fichiers (ou dossiers) donnés. `dossier@0.2` n'en garde que 20 %, par
    blocs de positions consécutives (des parties entières, à peu près).

    Renvoie les enregistrements et le numéro de bloc de chacun (pour séparer la validation)."""
    rng = np.random.default_rng(seed)
    parts, blocks = [], []
    next_block = 0
    for spec in paths:
        path, _, share = str(spec).partition("@")
        share = float(share) if share else 1.0
        p = Path(path)
        for f in sorted(p.rglob("*.bin")) if p.is_dir() else [p]:
            dtype = RECORD if f.name.endswith(".v2.bin") else RECORD_V1
            count = f.stat().st_size // dtype.itemsize
            recs = np.fromfile(f, dtype=dtype, count=count)
            if dtype is RECORD_V1:
                full = np.zeros(count, dtype=RECORD)
                for name in RECORD_V1.names:
                    full[name] = recs[name]
                full["qgain"] = np.nan
                recs = full
            ids = np.arange(count) // BLOCK + next_block
            next_block = int(ids[-1]) + 1 if count else next_block
            if share < 1.0:
                keep_blocks = rng.random(next_block) < share
                mask = keep_blocks[ids]
                recs, ids = recs[mask], ids[mask]
            parts.append(recs)
            blocks.append(ids)
    if not parts:
        return np.zeros(0, dtype=RECORD), np.zeros(0, dtype=np.int64)
    return np.concatenate(parts), np.concatenate(blocks)
