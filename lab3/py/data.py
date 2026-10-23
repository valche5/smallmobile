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
# propriétaires des régions présentes). La fenêtre entière tient alors en RAM : lue au hasard dans
# les fichiers mappés, elle faisait attendre l'apprentissage 80 ms par lot de 512 (lab3/journal.md).
PACKED = ".packed.npz"
# les octets des régions puis ceux de la partie, en un seul vecteur creux
SPARSE = REGIONS * NODE + GLOBAL


def pack(f):
    """La forme compacte du fichier de positions `f` (faite une fois, puis relue)."""
    out = f.with_name(f.name + PACKED)
    if out.exists() and out.stat().st_mtime >= f.stat().st_mtime:
        return out
    r = np.fromfile(f, dtype=RECORD)
    n = len(r)
    counts = games(f, n)
    keys = np.repeat(map_keys(r[np.cumsum(counts) - counts]), counts) if n else np.empty(0, np.uint64)
    present = np.arange(REGIONS)[None, :] < r["regions"][:, None].astype(np.int64)
    dense = np.concatenate([r["node"].reshape(n, -1), r["glob"]], 1)
    pos, col = np.nonzero(dense)
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
            adj=r["adj"][present],
            nz_count=np.bincount(pos, minlength=n).astype(np.uint16),
            nz_idx=col.astype(np.uint16),
            nz_val=dense[pos, col],
            pol_count=np.bincount(lpos, minlength=n).astype(np.uint16),
            pol_key=key.astype(np.uint16),
            pol_val=r["policy"][lpos, key],
        )
    tmp.rename(out)
    return out


def _segments(begin, count, sel):
    """Les éléments des positions `sel` dans un tableau à la suite (`begin`, `count` par position) :
    leur indice, la position de chacun dans `sel`, et son rang dans sa position."""
    c = count[sel].astype(np.int64)
    first = np.cumsum(c) - c
    rank = np.arange(int(c.sum())) - np.repeat(first, c)
    return np.repeat(begin[sel], c) + rank, np.repeat(np.arange(len(sel)), c), rank


class Packed:
    """Une fenêtre sous forme compacte, en RAM (`pack`) ; `take` redonne les champs pleins de
    `RECORD` d'un lot, `packed["weight"]` les petits champs d'une position."""

    SMALL = ("players", "regions", "weight", "win", "gain", "keys")
    SEGMENTED = (("owner", "regions"), ("adj", "regions"), ("nz_idx", "nz_count"), ("nz_val", "nz_count"),
                 ("pol_key", "pol_count"), ("pol_val", "pol_count"))

    def __init__(self, files):
        parts = [np.load(f) for f in files]
        small = {k: np.concatenate([p[k] for p in parts]) for k in self.SMALL + ("nz_count", "pol_count")}
        self.__dict__.update(small)
        # les grands tableaux remplis fichier par fichier : pas de copie entière en plus
        for name, _ in self.SEGMENTED:
            total = sum(p[name].shape[0] for p in parts)
            first = parts[0][name] if parts else np.empty(0, np.uint8)
            arr = np.empty((total,) + first.shape[1:], first.dtype)
            at = 0
            for p in parts:
                a = p[name]
                arr[at : at + len(a)] = a
                at += len(a)
                del a
            setattr(self, name, arr)
        def begin(count):
            c = count.astype(np.int64)
            return np.cumsum(c) - c
        self.rows_at, self.nz_at, self.pol_at = begin(self.regions), begin(self.nz_count), begin(self.pol_count)

    def __len__(self):
        return len(self.players)

    def __getitem__(self, name):
        return getattr(self, name)

    def take(self, sel):
        """Les champs pleins (`RECORD`) des positions `sel`."""
        n = len(sel)
        out = {k: getattr(self, k)[sel] for k in ("players", "regions", "weight", "win", "gain")}
        ids, b, rank = _segments(self.rows_at, self.regions, sel)
        out["owner"] = np.full((n, REGIONS), OFF_MAP, np.uint8)
        out["owner"][b, rank] = self.owner[ids]
        out["adj"] = np.zeros((n, REGIONS), np.uint64)
        out["adj"][b, rank] = self.adj[ids]
        ids, b, _ = _segments(self.nz_at, self.nz_count, sel)
        dense = np.zeros((n, SPARSE), np.uint8)
        dense[b, self.nz_idx[ids]] = self.nz_val[ids]
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
