"""Le score d'une arène (`lab arena --out r.tsv`), avec des intervalles qui respectent les donnes.

    python3 score.py r.tsv [--role R]          le score de chaque rôle (ou du rôle R)
    python3 score.py a.tsv b.tsv [--role R]    la différence A − B du rôle R (0 par défaut)
    python3 score.py r.tsv --role R --versus V la différence entre deux rôles de la même table

Score : écart des victoires à l'attendu à force égale (1/n), en points, moyenne à poids égaux des
tailles de table. L'unité indépendante est la donne : ses parties (une par rotation) partagent la
carte, les pioches et les dés. Chaque donne donne donc une valeur, la moyenne des écarts des places
du rôle ; les intervalles à 95 % viennent de la dispersion de ces valeurs. Deux arènes de même graine
jouent les mêmes donnes : leur différence est appariée donne par donne, comme celle de deux rôles
d'une même table.

Bibliothèque standard seulement : tourne aussi sur mini.
"""

import argparse
import math
import re
import sys
from collections import defaultdict

Z = 1.96


def read(path):
    """La graine, les noms des rôles et, par rôle, {(joueurs, donne): [lignes]}."""
    seed, names, rows = None, {}, defaultdict(lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if m := re.match(r"# graine (\d+)", line):
                    seed = int(m.group(1))
                elif m := re.match(r"# rôle (\d+) : (.*)", line):
                    names[int(m.group(1))] = m.group(2)
                continue
            if header is None:
                header = line.split("\t")
                continue
            r = dict(zip(header, line.split("\t")))
            n = int(r["players"])
            rows[int(r["role"])][(n, int(r["deal"]))].append(
                dict(
                    win=float(r["win"]),
                    margin=int(r["margin"]),
                    rank=int(r["rank"]),
                    decisions=int(r["decisions"]),
                    seconds=float(r["seconds"]),
                )
            )
    return seed, names, rows


def per_deal(rows):
    """Par taille : {donne: écart moyen des places du rôle à 1/n, en points}."""
    out = defaultdict(dict)
    for (n, deal), seats in rows.items():
        out[n][deal] = 100 * sum(s["win"] - 1 / n for s in seats) / len(seats)
    return out


def mean_var(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else float("nan")
    return m, v


def combine(sizes):
    """Moyenne à poids égaux de (moyenne, variance, donnes) par taille : score et demi-largeur à 95 %."""
    s = len(sizes)
    score = sum(m for m, _, _ in sizes) / s
    se = math.sqrt(sum(v / d for _, v, d in sizes)) / s
    return score, Z * se


def needed(variances, half_width):
    """Donnes par taille pour une demi-largeur `half_width` (points) sur le score à poids égaux."""
    s = len(variances)
    return math.ceil((Z / s) ** 2 * sum(variances) / half_width**2)


def describe(name, rows):
    deals = per_deal(rows)
    print(name)
    print("joueurs  donnes  victoires  écart à 1/n (95 %)  écart-type par donne  avance  place")
    sizes = []
    for n in sorted(deals):
        xs = list(deals[n].values())
        m, v = mean_var(xs)
        seats = [s for (k, _), ss in rows.items() if k == n for s in ss]
        win = 100 * sum(s["win"] for s in seats) / len(seats)
        margin = sum(s["margin"] for s in seats) / len(seats)
        rank = 1 + sum(s["rank"] for s in seats) / len(seats)
        print(
            f"{n:>7}  {len(xs):>6}  {win:>7.1f} %  {m:>+7.1f} ± {Z * math.sqrt(v / len(xs)):>4.1f}"
            f"  {math.sqrt(v):>20.1f}  {margin:>+6.1f}  {rank:>5.2f}"
        )
        sizes.append((m, v, len(xs)))
    score, hw = combine(sizes)
    seats = [s for ss in rows.values() for s in ss]
    old = 100 * sum(s["win"] - 1 / n for (n, _), ss in rows.items() for s in ss) / len(seats)
    decisions = sum(s["decisions"] for s in seats)
    ms = 1e3 * sum(s["seconds"] for s in seats) / max(decisions, 1)
    print(f"score (tailles à poids égaux) : {score:+.1f} ± {hw:.1f} points")
    print(f"ancien écart (pondéré par les parties) : {old:+.1f} points ; {ms:.1f} ms par décision")
    variances = [v for _, v, _ in sizes]
    print(f"donnes par taille pour ±1 point sur ce score : {needed(variances, 1.0)}")
    print()


def compare(a, b, role_a, role_b):
    (seed_a, names_a, rows_a), (seed_b, names_b, rows_b) = a, b
    da, db = per_deal(rows_a[role_a]), per_deal(rows_b[role_b])
    paired = seed_a is not None and seed_a == seed_b
    print(f"A : {names_a.get(role_a, role_a)}")
    print(f"B : {names_b.get(role_b, role_b)}")
    print("différence A − B, " + ("appariée par donne (même graine)" if paired else "graines différentes : non appariée"))
    print("joueurs  donnes  A − B (95 %)  corrélation  écart-type par donne")
    sizes = []
    for n in sorted(set(da) & set(db)):
        if paired:
            common = sorted(set(da[n]) & set(db[n]))
            diffs = [da[n][d] - db[n][d] for d in common]
            m, v = mean_var(diffs)
            ma, va = mean_var([da[n][d] for d in common])
            mb, vb = mean_var([db[n][d] for d in common])
            cov = sum((da[n][d] - ma) * (db[n][d] - mb) for d in common) / (len(common) - 1)
            corr = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")
            count = len(common)
        else:
            ma, va = mean_var(list(da[n].values()))
            mb, vb = mean_var(list(db[n].values()))
            count = min(len(da[n]), len(db[n]))
            # variance d'une différence non appariée, rapportée au nombre de donnes de chaque côté
            m, v = ma - mb, (va / len(da[n]) + vb / len(db[n])) * count
            corr = float("nan")
        print(f"{n:>7}  {count:>6}  {m:>+6.1f} ± {Z * math.sqrt(v / count):>4.1f}  {corr:>11.2f}  {math.sqrt(v):>20.1f}")
        sizes.append((m, v, count))
    if not sizes:
        sys.exit("aucune taille commune")
    score, hw = combine(sizes)
    verdict = "A meilleur" if score - hw > 0 else "B meilleur" if score + hw < 0 else "indécis"
    print(f"score A − B (tailles à poids égaux) : {score:+.1f} ± {hw:.1f} points ({verdict})")
    variances = [v for _, v, _ in sizes]
    print(f"donnes par taille pour ±1 point sur cette différence : {needed(variances, 1.0)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--role", type=int)
    ap.add_argument("--versus", type=int, help="un autre rôle de la même table")
    args = ap.parse_args()
    if len(args.files) == 1 and args.versus is not None:
        run = read(args.files[0])
        compare(run, run, args.role or 0, args.versus)
    elif len(args.files) == 1:
        _, names, rows = read(args.files[0])
        for role in sorted(rows) if args.role is None else [args.role]:
            describe(f"rôle {role} : {names.get(role, '?')}", rows[role])
    elif len(args.files) == 2:
        compare(read(args.files[0]), read(args.files[1]), args.role or 0, args.role or 0)
    else:
        sys.exit("un ou deux fichiers")


if __name__ == "__main__":
    main()
