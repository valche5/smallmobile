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


def load(paths, seed=0, with_value=False):
    """Les enregistrements des fichiers (ou dossiers) donnés. `dossier@0.2` n'en garde que 20 %, par
    blocs de positions consécutives (des parties entières, à peu près) ; `dossier:policy` ne sert
    qu'à la politique (sa valeur n'est pas apprise).

    Renvoie les enregistrements, le numéro de bloc de chacun (pour séparer la validation) et, si
    `with_value`, si sa valeur s'apprend."""
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
    recs = np.empty(total, dtype=RECORD)
    blocks = np.empty(total, dtype=np.int64)
    values = np.empty(total, dtype=bool)
    at = 0
    for f, dtype, count, mask, ids, value in plan:
        src = np.memmap(f, dtype=dtype, mode="r", shape=(count,)) if count else np.zeros(0, dtype=dtype)
        if mask is not None:
            src = src[mask]
        dst = recs[at : at + len(ids)]
        if dtype is RECORD:
            dst[...] = src
        else:
            for name in RECORD_V1.names:
                dst[name] = src[name]
            dst["qgain"] = np.nan
        blocks[at : at + len(ids)] = ids
        values[at : at + len(ids)] = value
        at += len(ids)
        del src
    return (recs, blocks, values) if with_value else (recs, blocks)
