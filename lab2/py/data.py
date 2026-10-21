"""Les parties d'entraînement écrites par `lab2 selfplay` (voir lab2/src/selfplay.rs pour le format)."""

from pathlib import Path

import numpy as np

KEYS = 608
REGIONS = 48
NODE = 77
# entrées de la partie (lab2/brain/src/encode.rs) : les 700 premières décrivent la position, les suivantes
# l'histoire des joueurs
GLOBAL = 820
ILLEGAL = 65535

RECORD = np.dtype(
    [
        ("players", "u1"),
        ("regions", "u1"),
        ("win", "<f4", 5),
        ("gain", "<f4", 5),
        # politique améliorée de la recherche, à une constante près (ILLEGAL : coup impossible)
        ("policy", "<u2", KEYS),
        ("adj", "<u8", REGIONS),
        ("node", "u1", (REGIONS, NODE)),
        ("glob", "u1", GLOBAL),
    ]
)
SUFFIX = ".pos"
# fichier `x.pos.games` à côté de `x.pos` : le nombre de positions de chaque partie, dans l'ordre
# (u32 petit-boutiste)
GAMES = ".games"


def games(f, count):
    """Le nombre de positions de chaque partie du fichier `f`, dans l'ordre (`f.games`)."""
    counts = np.fromfile(f.with_name(f.name + GAMES), dtype="<u4").astype(np.int64)
    assert counts.sum() == count, f"{f} : {counts.sum()} positions annoncées, {count} présentes"
    return counts[counts > 0]


def _mix(h):
    """Le mélange final de SplitMix64, sur des entiers 64 bits (modulo 2^64)."""
    with np.errstate(over="ignore"):
        h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return h ^ (h >> np.uint64(31))


def map_keys(first):
    """Une clé stable par partie, d'après sa carte (`first` : un enregistrement par partie). Elle ne
    dépend ni des fichiers ni de leur ordre : la même donne a la même clé partout."""
    h = _mix(first["players"].astype(np.uint64) << np.uint64(8) | first["regions"].astype(np.uint64))
    for j in range(first["adj"].shape[1]):
        h = _mix(h ^ first["adj"][:, j])
    return h


def in_validation(keys, share):
    """Les parties réservées à la validation : une règle fixe sur leur clé, la même pour toutes les
    générations (un réseau entraîné avec la même part n'a jamais vu ces parties)."""
    return keys % np.uint64(10_000) < np.uint64(round(share * 10_000))


def load(paths, keep=None):
    """Les enregistrements des fichiers (ou dossiers) donnés, en un seul tableau, et la clé de la
    partie de chacun (`map_keys`, pour séparer la validation). `keep` : ne garder que certaines
    parties (fonction des clés qui rend un masque, comme `in_validation`)."""
    files = []
    for path in paths:
        p = Path(path)
        files += sorted(p.rglob("*" + SUFFIX)) if p.is_dir() else [p]
    plan = []
    for f in files:
        count = f.stat().st_size // RECORD.itemsize
        counts = games(f, count)
        m = np.memmap(f, dtype=RECORD, mode="r", shape=(count,)) if count else np.zeros(0, dtype=RECORD)
        keys = np.repeat(map_keys(m[np.cumsum(counts) - counts]), counts)
        del m
        plan.append((f, count, keys, None if keep is None else keep(keys)))
    total = sum(len(keys) if sel is None else int(sel.sum()) for _, _, keys, sel in plan)
    recs = np.empty(total, dtype=RECORD)
    out = np.empty(total, dtype=np.uint64)
    at = 0
    for f, count, keys, sel in plan:
        if sel is not None:
            keys = keys[sel]
        if len(keys):
            m = np.memmap(f, dtype=RECORD, mode="r", shape=(count,))
            recs[at : at + len(keys)] = m if sel is None else m[sel]
            out[at : at + len(keys)] = keys
            at += len(keys)
            del m
    return recs, out
