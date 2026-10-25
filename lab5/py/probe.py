"""Analyse des confirmations de lab5, bibliothèque standard seulement.

Une observation = une position d'une partie indépendante. Les coups ont été sélectionnés sur
d'autres suites. Le gain conserve les valeurs négatives et les positions où le témoin est choisi.
Les intervalles rééchantillonnent les positions entières, séparément par taille et tiers.
Ce gain local n'est ni un taux de victoire en arène ni une distance au jeu optimal.
"""
import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

BANDS = ("early", "middle", "late")
MASK = (1 << 64) - 1


def mix(a, b):
    z = ((((a * 0x2545F4914F6CDD1D) & MASK) ^ b ^ 0x5851F42D4C957F2D) + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def world(seed, position, stage, sample):
    domain = {"select": 0x73656C656374, "confirm": 0x636F6E6669726D}[stage]
    return mix(mix(mix(mix(seed, position), domain), sample), 0)


def trials(path, keys, count, seed, position, stage):
    with open(path) as f:
        first = f.readline().rstrip("\n").split("\t")
        if len(first) != 2 or first[0] != "# selected":
            raise ValueError(f"{path}: sélection absente")
        selected = int(first[1])
        rows = {}
        for r in csv.DictReader(f, delimiter="\t"):
            s, key = int(r["sample"]), int(r["key"])
            win = float(r["win"])
            if not (0 <= s < count and key in keys and 0 <= win <= 1):
                raise ValueError(f"{path}: suite invalide")
            if (s, key) in rows or int(r["world"]) != world(seed, position, stage, s):
                raise ValueError(f"{path}: doublon ou graine incompatible")
            rows[s, key] = win
    if len(rows) != len(keys) * count or selected not in keys:
        raise ValueError(f"{path}: suites incomplètes")
    return selected, rows


def choose(baseline, keys, rows):
    totals = {key: sum(v for (_, k), v in rows.items() if k == key) for key in keys}
    best = baseline
    for key in sorted(keys):
        if totals[key] > totals[best]:
            best = key
    return best


def load(directory):
    directory = Path(directory)
    manifest = dict(line.rstrip("\n").split("\t", 1) for line in (directory / "manifest.tsv").read_text().splitlines())
    if manifest["format"] != "lab5-probe-1":
        raise ValueError("format inconnu")
    players = [int(x) for x in manifest["players"].split(",")]
    per = int(manifest["positions_per_cell"])
    total = len(players) * 3 * per
    seed, select_n, confirm_n = (int(manifest[k]) for k in ("seed", "select", "confirm"))
    results = []
    for i in range(total):
        prefix = directory / f"p{i:06}"
        with open(str(prefix) + ".position.tsv") as f:
            p = next(csv.DictReader(f, delimiter="\t"))
        n, band = players[i % len(players)], BANDS[(i // len(players)) % 3]
        if int(p["id"]) != i or int(p["players"]) != n or p["band"] != band:
            raise ValueError(f"position {i}: mauvaise strate")
        baseline = int(p["baseline"])
        keys = [int(x) for x in p["candidates"].split(",")]
        if len(set(keys)) != len(keys) or baseline not in keys:
            raise ValueError(f"position {i}: candidats invalides")
        selected, selection = trials(str(prefix) + ".select.tsv", keys, select_n, seed, i, "select")
        if selected != choose(baseline, keys, selection):
            raise ValueError(f"position {i}: sélection modifiée")
        checked = list(dict.fromkeys((baseline, selected)))
        frozen, confirmation = trials(str(prefix) + ".confirm.tsv", checked, confirm_n, seed, i, "confirm")
        if frozen != selected:
            raise ValueError(f"position {i}: choix changé pendant la confirmation")
        delta = [confirmation[s, selected] - confirmation[s, baseline] for s in range(confirm_n)]
        gain = mean(delta)
        results.append(dict(id=i, players=n, band=band, phase=p["phase"], baseline=baseline, selected=selected,
                            gain=gain, before=mean(confirmation[s, baseline] for s in range(confirm_n)),
                            after=mean(confirmation[s, selected] for s in range(confirm_n)), samples=confirm_n))
    return manifest, results


def estimate(groups, repetitions=4000):
    """Moyenne à poids égaux des cellules ; bootstrap par partie source, sans tronquer les gains."""
    cells = list(groups.values())
    score = mean(mean(cell) for cell in cells)
    if any(len(cell) < 2 for cell in cells):
        return score, None
    rng = random.Random(51005)
    draws = sorted(mean(mean(rng.choices(cell, k=len(cell))) for cell in cells) for _ in range(repetitions))
    return score, (draws[int(.025 * repetitions)], draws[min(repetitions - 1, math.ceil(.975 * repetitions) - 1)])


def report(manifest, rows, repetitions=4000):
    print("Gain local de victoire : coup sélectionné moins coup de c130, suites de confirmation uniquement.")
    print("Ce résultat mesure un changement de coup suivi par c130 ; ce n'est pas un gain en arène.")
    print(f"{len(rows)} parties sources indépendantes ; {manifest['select']} suites de sélection par candidat, "
          f"{manifest['confirm']} de confirmation par coup.")
    print("joueurs  tiers      positions  coups changés  gain (points)  intervalle bootstrap 95 %")
    groups = defaultdict(list)
    for r in rows:
        groups[r["players"], r["band"]].append(r["gain"])
    def line(label, selected, cells):
        score, ci = estimate(cells, repetitions)
        interval = "non estimé (pilote)" if ci is None else f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"
        changed = sum(r["selected"] != r["baseline"] for r in selected)
        print(f"{label:18}  {len(selected):9}  {changed:12}  {100 * score:+13.2f}  {interval}")
    for n, band in sorted(groups, key=lambda k: (k[0], BANDS.index(k[1]))):
        subset = [r for r in rows if (r["players"], r["band"]) == (n, band)]
        line(f"{n}        {band}", subset, {(n, band): groups[n, band]})
    line("global équilibré", rows, groups)
    print("\nPhases observées (descriptif, fréquences dépendantes du protocole de tirage) :")
    for phase in sorted({r["phase"] for r in rows}):
        subset = [r for r in rows if r["phase"] == phase]
        print(f"{phase}: {len(subset)} positions, gain moyen {100 * mean(r['gain'] for r in subset):+.2f} points")
    print("\nPositions à inspecter (classement exploratoire, aucune significativité individuelle revendiquée) :")
    for r in sorted(rows, key=lambda r: abs(r["gain"]), reverse=True)[:12]:
        print(f"p{r['id']:06}: {r['players']} j. {r['band']} {r['phase']}, "
              f"coup {r['baseline']} → {r['selected']}, {100*r['before']:.1f} → {100*r['after']:.1f} %, "
              f"écart {100*r['gain']:+.1f} points")
    print("\nUne absence de gain ne prouve pas l'optimalité : candidats limités, puis continuation par c130.")
    print("Les intervalles sont approximatifs ; un pilote à peu de positions sert à mesurer le coût.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory")
    ap.add_argument("--bootstrap", type=int, default=4000)
    args = ap.parse_args()
    if args.bootstrap < 100:
        ap.error("au moins 100 réplications bootstrap")
    try:
        manifest, rows = load(args.directory)
        report(manifest, rows, args.bootstrap)
    except (ValueError, KeyError, OSError, StopIteration) as e:
        ap.exit(2, f"lot incomplet ou invalide : {e}\n")


if __name__ == "__main__":
    main()
