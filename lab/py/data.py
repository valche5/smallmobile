"""Les parties d'entraînement écrites par `lab selfplay` (voir lab/src/selfplay.rs pour le format)."""

from pathlib import Path

import numpy as np

KEYS = 448
REGIONS = 48
NODE = 77
GLOBAL = 700
ILLEGAL = 65535

RECORD = np.dtype(
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


def load(paths) -> np.ndarray:
    """Tous les enregistrements des fichiers (ou dossiers) donnés."""
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.bin")) if p.is_dir() else [p]
    parts = []
    for f in files:
        size = f.stat().st_size
        if size % RECORD.itemsize:
            # un fichier en cours d'écriture : on ne garde que les enregistrements complets
            print(f"{f} : fin incomplète ignorée")
        parts.append(np.fromfile(f, dtype=RECORD, count=size // RECORD.itemsize))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=RECORD)
