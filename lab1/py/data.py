"""Les parties d'entraînement écrites par `lab1 selfplay` et `lab1 judge` (voir lab1/src/selfplay.rs pour le
format)."""

from pathlib import Path

import numpy as np

KEYS = 448
REGIONS = 48
NODE = 77
# entrées de la partie (lab1/brain/src/encode.rs) : les 700 premières décrivent la position, les suivantes
# l'histoire des joueurs
GLOBAL = 820
ILLEGAL = 65535
# drapeaux d'un enregistrement (champ `flags`) : table variée ; coup observé d'un autre joueur que
# l'IA qui apprend, qui n'apprend que la tête « coup réel »
MIXED, OBSERVED = 1, 2

RECORD = np.dtype(
    [
        ("players", "u1"),
        ("regions", "u1"),
        ("full", "u1"),
        ("flags", "u1"),
        ("win", "<f4", 5),
        ("gain", "<f4", 5),
        ("visits", "<u2", KEYS),
        ("adj", "<u8", REGIONS),
        ("node", "u1", (REGIONS, NODE)),
        ("glob", "u1", GLOBAL),
        # la clé du coup joué (ILLEGAL : inconnu)
        ("played", "<u2"),
    ]
)
SUFFIX = ".v6.bin"
# fichier `x.v6.bin.games` à côté de `x.v6.bin` : le nombre de positions de chaque partie, dans
# l'ordre (u32 petit-boutiste)
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


def load(paths):
    """Les enregistrements des fichiers (ou dossiers) donnés, en un seul tableau, et la clé de la
    partie de chacun (`map_keys`, pour séparer la validation)."""
    files = []
    for path in paths:
        p = Path(path)
        files += sorted(p.rglob("*" + SUFFIX)) if p.is_dir() else [p]
    sizes = [f.stat().st_size // RECORD.itemsize for f in files]
    recs = np.empty(sum(sizes), dtype=RECORD)
    keys = np.empty(sum(sizes), dtype=np.uint64)
    at = 0
    for f, count in zip(files, sizes):
        counts = games(f, count)
        m = np.memmap(f, dtype=RECORD, mode="r", shape=(count,)) if count else np.zeros(0, dtype=RECORD)
        recs[at : at + count] = m
        keys[at : at + count] = np.repeat(map_keys(m[np.cumsum(counts) - counts]), counts)
        at += count
        del m
    return recs, keys
