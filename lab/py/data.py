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


# fichier `x.bin.games` à côté de `x.bin` : le nombre de positions de chaque partie, dans l'ordre
# (u32 petit-boutiste), écrit par `lab selfplay` et `lab label`, ou retrouvé d'après les cartes
GAMES = ".games"


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


def games(f, dtype, count):
    """Le nombre de positions de chaque partie du fichier `f`, dans l'ordre. Lu dans `f.games` ; à
    défaut, retrouvé d'après les cartes (les positions d'une partie se suivent et partagent la même
    carte, qui change d'une donne à l'autre), puis écrit dans `f.games` si possible."""
    side = f.with_name(f.name + GAMES)
    if side.exists():
        counts = np.fromfile(side, dtype="<u4").astype(np.int64)
        if counts.sum() == count:
            return counts[counts > 0]
    m = np.memmap(f, dtype=dtype, mode="r", shape=(count,)) if count else np.zeros(0, dtype=dtype)
    starts = [0] if count else []
    prev = None
    for a in range(0, count, 1 << 18):
        chunk = m[a : a + (1 << 18)]
        key = np.concatenate(
            [chunk["players"][:, None].astype(np.uint64), chunk["regions"][:, None].astype(np.uint64), chunk["adj"]],
            axis=1,
        )
        if prev is not None:
            key = np.concatenate([prev[None], key])
        change = np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1
        starts.extend((change + a - (prev is not None)).tolist())
        prev = key[-1]
    counts = np.diff(np.array(starts + [count], dtype=np.int64))
    try:
        counts.astype("<u4").tofile(side)
    except OSError:
        pass
    return counts


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


def load(paths, seed=0, with_value=False, lazy=False):
    """Les enregistrements des fichiers (ou dossiers) donnés. `dossier@0.2` n'en garde que 20 %, par
    parties entières ; `dossier:policy` ne sert qu'à la politique (sa valeur n'est pas apprise).

    Renvoie les enregistrements, la clé de la partie de chacun (`map_keys`, pour séparer la
    validation) et, si `with_value`, si sa valeur s'apprend. `lazy` : les enregistrements restent
    dans les fichiers (`Records`), lus à chaque lot."""
    rng = np.random.default_rng(seed)
    # premier passage : quels enregistrements garder (sans tout lire), pour tout écrire d'un coup
    # dans un seul tableau (pas de copie : les données peuvent occuper la moitié de la mémoire)
    plan = []
    for spec in paths:
        spec, _, tag = str(spec).partition(":")
        path, _, share = spec.partition("@")
        share = float(share) if share else 1.0
        p = Path(path)
        for f in sorted(p.rglob("*.bin")) if p.is_dir() else [p]:
            dtype = RECORD if f.name.endswith(".v2.bin") else RECORD_V1
            count = f.stat().st_size // dtype.itemsize
            counts = games(f, dtype, count)
            starts = np.cumsum(counts) - counts
            m = np.memmap(f, dtype=dtype, mode="r", shape=(count,)) if count else np.zeros(0, dtype=dtype)
            ids = np.repeat(map_keys(m[starts]), counts)
            del m
            mask = None
            if share < 1.0:
                mask = np.repeat(rng.random(len(counts)) < share, counts)
                ids = ids[mask]
            plan.append((f, dtype, count, mask, ids, tag != "policy"))
    total = sum(len(ids) for *_, ids, _ in plan)
    keys = np.concatenate([ids for *_, ids, _ in plan]) if plan else np.zeros(0, np.uint64)
    values = np.concatenate([np.full(len(ids), v) for *_, ids, v in plan]) if plan else np.zeros(0, bool)
    if lazy:
        sources = [
            (f, dtype, count, np.arange(count) if mask is None else np.flatnonzero(mask))
            for f, dtype, count, mask, _, _ in plan
        ]
        recs = Records(sources)
        return (recs, keys, values) if with_value else (recs, keys)
    recs = np.empty(total, dtype=RECORD)
    at = 0
    for f, dtype, count, mask, ids, _ in plan:
        src = np.memmap(f, dtype=dtype, mode="r", shape=(count,)) if count else np.zeros(0, dtype=dtype)
        if mask is not None:
            src = src[mask]
        fill(recs[at : at + len(ids)], src)
        at += len(ids)
        del src
    return (recs, keys, values) if with_value else (recs, keys)
