"""Les parties d'entraînement écrites par `lab3 selfplay` (voir lab3/src/selfplay.rs pour le format)."""

import mmap
import os
import struct
import zipfile
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
# fichier `x.pos.values` : la part de victoire attendue de chaque joueur relatif selon la recherche,
# par position (f32 × 5 ; lab3/src/selfplay.rs, `values_file`). Absent des cycles d'avant le
# 24 octobre : la valeur y est le résultat de la partie (`win`).
VALUES = ".values"


# fenêtre (en positions) quand `total` positions ont été jouées : tout jusqu'à WINDOW_MIN, puis
# WINDOW_MIN × (1 + WINDOW_GROWTH × ((total / WINDOW_MIN)^WINDOW_POWER − 1) / WINDOW_POWER) ; la
# forme de KataGo. Minimum : quatre cycles de 25 000 parties à 4 joueurs (environ 990 000 positions
# chacun), pour qu'un pas, qui tire USES = 4 fois les positions de son dernier cycle
# (train.py), fasse au plus une passe sur sa fenêtre ; le réseau de départ joue jusqu'à la remplir
# (`filled`). Avec une fenêtre d'un seul cycle, le c060 de lab2 continué à 4 joueurs perdait 8 à 9
# points en trois pas (lab3/journal.md) : quatre passes par pas sur les mêmes 25 000 parties.
WINDOW_MIN = 4_000_000
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


def filled(cycles, last):
    """Si la fenêtre minimale est remplie au cycle `last` : un cycle de plus, de la taille de `last`,
    n'y tiendrait plus (au-delà de WINDOW_MIN, la fenêtre laisse sortir les plus anciens)."""
    dirs = [p for p in Path(cycles).iterdir() if p.is_dir() and p.name <= last]
    return sum(positions(d) for d in dirs) + positions(Path(cycles) / last) > WINDOW_MIN


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


# La forme compacte d'un fichier de positions (`pack`), à côté de lui : ce qui n'est pas vide d'un
# enregistrement de 6 210 octets, ~1,9 Ko par position (octets non nuls des régions et de la partie,
# un indice de 16 bits et une valeur chacun ; politique des seuls coups possibles ; voisines et
# propriétaires des régions présentes). Lue au hasard dans les fichiers `.pos` mappés, la fenêtre
# faisait attendre l'apprentissage 80 ms par lot de 512 (lab3/journal.md) ; sous forme compacte,
# ses grands tableaux restent mappés dans ces fichiers (`_members`), et le cache du système en
# garde ce qui tient en RAM : la fenêtre grandit (~24 Go à c040) sans déborder les 31 Go du PC.
PACKED = ".packed3.npz"
# les octets des régions puis ceux de la partie, en un seul vecteur creux : où sont les non nuls (un
# bit chacun : 10 octets par région présente, `GLOBAL_MASK` pour la partie), puis leurs valeurs
SPARSE = REGIONS * NODE + GLOBAL
GLOBAL_MASK = (GLOBAL + 7) // 8


def pack(f):
    """La forme compacte du fichier de positions `f` (faite une fois, puis relue). Par position :
    ses régions présentes seulement, le masque de ses octets non nuls et leurs valeurs, ses coups
    possibles seulement ; les adjacences d'une même carte une seule fois (seuls 3 % des blocs
    diffèrent : la carte change peu d'une position à la suite). Les valeurs non nulles : un bit dit
    si elle vaut 1 (62 % d'entre elles), un octet seulement pour les autres. ~880 octets par position
    au lieu des 6 210 de `RECORD` (1 940 avec les index des non nuls et toutes les adjacences, 1 212
    avec le masque des 48 régions et un octet par valeur)."""
    out = f.with_name(f.name + PACKED)
    if out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
        return out
    r = np.fromfile(f, dtype=RECORD)
    _forget(f)
    n = len(r)
    counts = games(f, n)
    keys = np.repeat(map_keys(r[np.cumsum(counts) - counts]), counts) if n else np.empty(0, np.uint64)
    present = np.arange(REGIONS)[None, :] < r["regions"][:, None].astype(np.int64)
    dense = np.concatenate([r["node"].reshape(n, -1), r["glob"]], 1)
    nonzero = dense != 0
    node_nz = r["node"] != 0
    assert not node_nz[~present].any(), f"{f} : une région absente a des entrées"
    vals = dense[nonzero]
    count = nonzero.sum(1)
    big = vals != 1
    blocks, block_of, rows = {}, np.empty(n, np.uint32), []
    for i in range(n):
        row = r["adj"][i, : r["regions"][i]]
        block_of[i] = blocks.setdefault(row.tobytes(), len(blocks))
        if block_of[i] == len(rows):
            rows.append(row)
    values = f.with_name(f.name + VALUES)
    extra = {}
    if values.exists():
        extra["value"] = np.fromfile(values, dtype="<f4").reshape(r["win"].shape)
    legal = r["policy"] != ILLEGAL
    lpos, key = np.nonzero(legal)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            players=r["players"],
            regions=r["regions"],
            weight=r["weight"],
            win=r["win"],
            gain=r["gain"],
            keys=keys,
            owner=r["owner"][present],
            adj=np.concatenate(rows) if rows else np.empty(0, np.uint64),
            adj_count=np.array([len(row) for row in rows], np.uint8),
            adj_block=block_of,
            nz_count=count.astype(np.uint16),
            nz_rows=np.packbits(node_nz[present], axis=1),
            nz_glob=np.packbits(r["glob"] != 0, axis=1),
            nz_ones=np.packbits(~big),
            nz_big_count=np.bincount(np.repeat(np.arange(n), count)[big], minlength=n).astype(np.uint16),
            nz_big=vals[big],
            pol_count=np.bincount(lpos, minlength=n).astype(np.uint16),
            pol_key=key.astype(np.uint16),
            pol_val=r["policy"][lpos, key],
            **extra,
        )
    tmp.rename(out)
    _forget(out)
    return out


def _forget(f):
    """Sort `f` du cache du système : la fenêtre mappée (`Packed`) y garde sa place."""
    fd = os.open(f, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _members(path):
    """Les tableaux d'un `.npz` non compressé (`np.savez`), mappés à leur place dans le fichier.
    Toutes les pages sont installées à l'ouverture (`MAP_POPULATE`) : lues une fois à la suite,
    puis aucune faute de page pendant l'apprentissage (sans elle, 33 ms par lot au lieu de 2). Ce
    sont les pages du cache du système, qu'il reprend s'il manque de mémoire."""
    out = {}
    with zipfile.ZipFile(path) as z, open(path, "rb") as f:
        whole = mmap.mmap(f.fileno(), 0, flags=mmap.MAP_SHARED | mmap.MAP_POPULATE, prot=mmap.PROT_READ)
        for info in z.infolist():
            assert info.compress_type == zipfile.ZIP_STORED, f"{path} : compressé"
            f.seek(info.header_offset)
            name_len, extra_len = struct.unpack("<HH", f.read(30)[26:30])
            f.seek(info.header_offset + 30 + name_len + extra_len)
            version = np.lib.format.read_magic(f)
            read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = read(f)
            assert not fortran
            name = info.filename.removesuffix(".npy")
            count = int(np.prod(shape))
            out[name] = np.frombuffer(whole, dtype=dtype, count=count, offset=f.tell()).reshape(shape)
    return out


class Parts:
    """Les mêmes tableaux de plusieurs fichiers, à la suite, sans les copier : `parts[ids]`."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.starts = np.cumsum([0] + [len(a) for a in arrays])

    def __len__(self):
        return int(self.starts[-1])

    def __getitem__(self, ids):
        first = self.arrays[0]
        out = np.empty((len(ids),) + first.shape[1:], first.dtype)
        # par fichier, dans l'ordre des indices
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        cuts = np.searchsorted(sorted_ids, self.starts)
        for f, a in enumerate(self.arrays):
            lo, hi = cuts[f], cuts[f + 1]
            if hi > lo:
                out[order[lo:hi]] = a[sorted_ids[lo:hi] - self.starts[f]]
        return out


class Bits:
    """Les flots de bits de plusieurs fichiers (`np.packbits`), à la suite : `bits[ids]`, 0 ou 1."""

    def __init__(self, arrays, counts):
        self.arrays = arrays
        self.starts = np.cumsum([0] + list(counts))

    def __getitem__(self, ids):
        out = np.empty(len(ids), np.uint8)
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        cuts = np.searchsorted(sorted_ids, self.starts)
        for f, a in enumerate(self.arrays):
            lo, hi = cuts[f], cuts[f + 1]
            if hi > lo:
                local = sorted_ids[lo:hi] - self.starts[f]
                out[order[lo:hi]] = (a[local >> 3] >> (7 - (local & 7))) & 1
        return out


def _segments(begin, count, sel):
    """Les éléments des positions `sel` dans un tableau à la suite (`begin`, `count` par position) :
    leur indice, la position de chacun dans `sel`, et son rang dans sa position."""
    c = count[sel].astype(np.int64)
    first = np.cumsum(c) - c
    rank = np.arange(int(c.sum())) - np.repeat(first, c)
    return np.repeat(begin[sel], c) + rank, np.repeat(np.arange(len(sel)), c), rank


class Packed:
    """Une fenêtre sous forme compacte (`pack`) : ses petits champs en RAM (une soixantaine d'octets
    par position), les grands mappés dans les fichiers (`Parts`) ; `take` redonne les champs pleins
    de `RECORD` d'un lot, `packed["weight"]` les petits champs d'une position."""

    SMALL = ("players", "regions", "weight", "win", "gain", "keys")
    MAPPED = ("owner", "adj", "nz_rows", "nz_glob", "nz_big", "pol_key", "pol_val")

    def __init__(self, files):
        parts = [_members(f) for f in files]
        small = {k: np.concatenate([p[k] for p in parts]) for k in self.SMALL + ("nz_count", "nz_big_count", "pol_count")}
        # la valeur de la recherche, ou le résultat de la partie pour un fichier qui ne l'a pas
        small["value"] = np.concatenate([p.get("value", p["win"]) for p in parts]) if parts else small["win"]
        self.__dict__.update(small)
        for name in self.MAPPED:
            setattr(self, name, Parts([p[name] for p in parts]))
        self.nz_ones = Bits([p["nz_ones"] for p in parts], [int(p["nz_count"].sum(dtype=np.int64)) for p in parts])
        def begin(count):
            c = count.astype(np.int64)
            return np.cumsum(c) - c
        self.rows_at, self.nz_at, self.pol_at = begin(self.regions), begin(self.nz_count), begin(self.pol_count)
        self.big_at = begin(self.nz_big_count)
        # le début du bloc d'adjacences de chaque position, dans les blocs de tous les fichiers
        ats, row0 = [], 0
        for p in parts:
            ats.append(row0 + begin(p["adj_count"])[p["adj_block"]])
            row0 += len(p["adj"])
        self.adj_at = np.concatenate(ats) if ats else np.empty(0, np.int64)

    def __len__(self):
        return len(self.players)

    def __getitem__(self, name):
        return getattr(self, name)

    def take(self, sel):
        """Les champs pleins (`RECORD`) des positions `sel`."""
        n = len(sel)
        out = {k: getattr(self, k)[sel] for k in ("players", "regions", "weight", "win", "gain", "value")}
        ids, b, rank = _segments(self.rows_at, self.regions, sel)
        out["owner"] = np.full((n, REGIONS), OFF_MAP, np.uint8)
        out["owner"][b, rank] = self.owner[ids]
        # où sont les non nuls : les lignes des régions présentes, puis la partie
        node_nz = np.zeros((n, REGIONS, NODE), bool)
        node_nz[b, rank] = np.unpackbits(self.nz_rows[ids], axis=1, count=NODE).view(bool)
        glob_nz = np.unpackbits(self.nz_glob[sel], axis=1, count=GLOBAL).view(bool)
        ids, b, rank = _segments(self.adj_at, self.regions, sel)
        out["adj"] = np.zeros((n, REGIONS), np.uint64)
        out["adj"][b, rank] = self.adj[ids]
        # leurs valeurs, dans l'ordre des non nuls, ligne par ligne : 1, ou la suivante des autres
        ids, _, _ = _segments(self.nz_at, self.nz_count, sel)
        vals = self.nz_ones[ids]
        big, _, _ = _segments(self.big_at, self.nz_big_count, sel)
        vals[vals == 0] = self.nz_big[big]
        dense = np.zeros((n, SPARSE), np.uint8)
        dense[np.concatenate([node_nz.reshape(n, -1), glob_nz], 1)] = vals
        out["node"] = np.ascontiguousarray(dense[:, : REGIONS * NODE]).reshape(n, REGIONS, NODE)
        out["glob"] = np.ascontiguousarray(dense[:, REGIONS * NODE :])
        ids, b, _ = _segments(self.pol_at, self.pol_count, sel)
        out["policy"] = np.full((n, KEYS), ILLEGAL, np.uint16)
        out["policy"][b, self.pol_key[ids]] = self.pol_val[ids]
        return out


def load_packed(paths):
    """La fenêtre des fichiers (ou dossiers) donnés, sous forme compacte (`Packed`), et la clé de la
    partie de chaque position (`map_keys`)."""
    files = []
    for path in paths:
        p = Path(path)
        files += sorted(p.rglob("*" + SUFFIX)) if p.is_dir() else [p]
    recs = Packed([pack(f) for f in files])
    return recs, recs.keys


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
