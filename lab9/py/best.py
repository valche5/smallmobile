"""Le meilleur réseau d'un apprentissage (`train.py`), d'après son journal : la validation de plus petite
somme politique + valeur finale, et le critère de la phase 2 (`lab9/lab.md`) : politique et valeur au
moins aussi bonnes que le réseau de c130 seul, en tout et par nombre de joueurs.

    python best.py ../runs/i1/train.log
"""

import re
import sys


def blocks(text):
    """Les validations : (lot, {partie : {mesure : valeur}})."""
    out = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "validation (" not in line:
            continue
        step = None
        for back in range(i - 1, -1, -1):
            m = re.match(r"lot (\d+)/", lines[back])
            if m:
                step = int(m.group(1))
                break
        parts = {"tout": line.split(":", 1)[1]}
        for extra in lines[i + 1 : i + 10]:
            m = re.match(r"\s+(\S+(?: j\.)?) : (.*)", extra)
            if m:
                parts[m.group(1)] = m.group(2)
        out.append((step, {k: dict(re.findall(r"(\w+) (-?[\d.]+)%?", v)) for k, v in parts.items()}))
    return out


def main():
    vals = blocks(open(sys.argv[1]).read())
    if not vals:
        print("aucune validation")
        return
    for step, parts in vals:
        t = parts["tout"]
        print(f"lot {step} : pol {t['pol']} (c130 {t['c130_pol']}), val {t['val']} (c130 {t['c130_val']}), précision {t['acc']} % (c130 {t['c130_acc']} %)")
    step, parts = min(vals, key=lambda v: float(v[1]["tout"]["pol"]) + float(v[1]["tout"]["val"]))
    print(f"\nmeilleur : lot {step}")
    ok = True
    for name in ["tout", "2 j.", "3 j.", "4 j.", "5 j."]:
        p = parts.get(name)
        if not p:
            continue
        pol, val = float(p["pol"]) <= float(p["c130_pol"]), float(p["val"]) <= float(p["c130_val"])
        ok &= pol and val
        print(f"  {name:5} politique {p['pol']} contre {p['c130_pol']} {'oui' if pol else 'non'}, "
              f"valeur {p['val']} contre {p['c130_val']} {'oui' if val else 'non'}")
    print(f"critère de la phase 2 : {'atteint' if ok else 'non atteint'}")


if __name__ == "__main__":
    main()
