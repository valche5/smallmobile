"""Les parties d'entraînement écrites par `lab3 selfplay` (voir lab3/src/selfplay.rs pour le format)."""

from pathlib import Path

import numpy as np

KEYS = 608
REGIONS = 48
NODE = 77
# entrées de la partie (lab3/brain/src/encode.rs) : les 700 premières décrivent la position, les suivantes
# l'histoire des joueurs
GLOBAL = 820
ILLEGAL = 65535
OFF_MAP = 255
# propriétaires possibles d'une région : personne, puis chaque joueur relatif
OWNERS = 6

RECORD = np.dtype(
    [
        ("players", "u1"),
        ("regions", "u1"),
        # poids de la position, 1 en moyenne : surprise de la recherche (lab3/src/selfplay.rs, `weights`)
        ("weight", "<f4"),
        ("win", "<f4", 5),
        ("gain", "<f4", 5),
        # à qui appartient chaque région au début du prochain tour du joueur qui agit : 0 personne,
        # 1 + joueur relatif, OFF_MAP hors de la carte
        ("owner", "u1", REGIONS),
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


# fenêtre (en positions) quand `total` positions ont été jouées : tout jusqu'à WINDOW_MIN, puis
# WINDOW_MIN × (1 + WINDOW_GROWTH × ((total / WINDOW_MIN)^WINDOW_POWER − 1) / WINDOW_POWER) ; la
# forme de KataGo ; au départ, le petit minimum de la première boucle de lab2 avec son petit réseau
# (choix de l'utilisateur, lab3/journal.md)
WINDOW_MIN = 250_000
WINDOW_GROWTH = 0.4
WINDOW_POWER = 0.75


def window(total):
    """Le nombre de positions de la fenêtre quand `total` positions ont été jouées."""
    if total <= WINDOW_MIN:
        return total
    x = (total / WINDOW_MIN) ** WINDOW_POWER
    return WINDOW_MIN * (1 + WINDOW_GROWTH * (x - 1) / WINDOW_POWER)


def positions(d):
    """Le nombre de positions du cycle `d`, d'après ses fichiers `.games` (gardés quand les positions
    d'un vieux cycle sont effacées du bucket, `lab3/prune.sh`)."""
    return sum(int(np.fromfile(f, dtype="<u4").sum()) for f in d.glob("*" + SUFFIX + GAMES))


def chosen(cycles, last):
    """Les cycles de la fenêtre jusqu'à `last`, du plus ancien au plus récent (les derniers qui y
    tiennent, au moins `last`), et le nombre de positions de `last`."""
    dirs = sorted(p for p in Path(cycles).iterdir() if p.is_dir() and p.name <= last)
    assert dirs and dirs[-1].name == last, f"cycle {last} absent de {cycles}"
    sizes = [positions(d) for d in dirs]
    room = window(sum(sizes))
    keep, count = [], 0
    for d, n in zip(reversed(dirs), reversed(sizes)):
        if keep and count + n > room:
            break
        keep.append(d)
        count += n
    return keep[::-1], sizes[-1]


def outside(cycles, margin=1):
    """Les cycles sortis de la fenêtre du dernier cycle, moins `margin` cycles juste avant elle : ceux
    dont les positions ne servent plus (la fenêtre avance avec les cycles)."""
    dirs = sorted(p for p in Path(cycles).iterdir() if p.is_dir())
    first = chosen(cycles, dirs[-1].name)[0][0]
    old = [p for p in dirs if p.name < first.name]
    return [p.name for p in old[: max(0, len(old) - margin)]]


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
