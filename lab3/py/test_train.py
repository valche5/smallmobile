"""Régressions du chargement des lots, et de l'apprentissage en continu sur une petite fenêtre.

    cd lab3/py && .venv/bin/python -m unittest test_train
"""

import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch

from data import GAMES, ILLEGAL, NODE, OFF_MAP, RECORD, REGIONS, load, load_packed
from model import Net, export, load as load_net
from train import batches, fields, main


def random_records(count, rng):
    """`count` positions au hasard, de tailles et de champs variés."""
    recs = np.zeros(count, dtype=RECORD)
    for i, r in enumerate(recs):
        n = int(rng.integers(5, REGIONS + 1))
        r["players"], r["regions"] = 2 + i % 4, n
        r["weight"] = rng.random()
        r["win"], r["gain"] = rng.random(5), rng.normal(size=5)
        r["owner"][:n], r["owner"][n:] = rng.integers(0, 6, n), OFF_MAP
        r["adj"][:n] = rng.integers(0, 2**63, n, dtype=np.uint64)
        node = rng.integers(0, 256, (n, NODE)) * (rng.random((n, NODE)) < 0.15)
        r["node"][:n] = node
        r["glob"] = rng.integers(0, 256, r["glob"].shape) * (rng.random(r["glob"].shape) < 0.15)
        r["policy"] = ILLEGAL
        keys = rng.choice(len(r["policy"]), int(rng.integers(1, 30)), replace=False)
        r["policy"][keys] = rng.integers(0, 60000, len(keys))
    return recs


class BatchesTest(unittest.TestCase):
    def setUp(self):
        self.recs = np.zeros(3, dtype=RECORD)
        self.recs["players"] = 4
        self.recs["regions"] = 3
        self.recs["policy"] = ILLEGAL
        self.recs["policy"][:, 0] = 10
        for i in range(3):
            self.recs["win"][i, i] = 1
        self.recs["gain"] = np.arange(15).reshape(3, 5)

    def test_single_position(self):
        b = next(batches(self.recs, np.array([2]), 512, "cpu"))
        torch.testing.assert_close(b["win"], torch.tensor([[0., 0., 1., 0., 0.]]))
        self.assertEqual(tuple(b["node"].shape), (1, 48, 77))
        self.assertEqual(b["policy"][0, 0].item(), 1.)

    def test_final_single_position(self):
        # Le dernier lot n'a qu'une position.
        bs = list(batches(self.recs, np.array([2, 0, 1]), 2, "cpu"))
        self.assertEqual([len(b["win"]) for b in bs], [2, 1])
        # Les indices sont triés dans chaque lot, pas dans toute la séquence.
        torch.testing.assert_close(bs[0]["win"], torch.from_numpy(self.recs["win"][[0, 2]].copy()))
        torch.testing.assert_close(bs[1]["win"], torch.from_numpy(self.recs["win"][[1]].copy()))

    def test_packed_gives_back_the_records(self):
        """La forme compacte redonne les mêmes champs, au bit près, sur plusieurs fichiers."""
        recs = random_records(40, np.random.default_rng(3))
        with tempfile.TemporaryDirectory() as tmp:
            for name, part, counts in (("a.pos", recs[:15], [7, 8]), ("b.pos", recs[15:], [10, 0, 15])):
                part.tofile(Path(tmp) / name)
                np.array(counts, "<u4").tofile(Path(tmp) / (name + GAMES))
            packed, keys = load_packed([tmp])
            self.assertEqual(len(packed), len(recs))
            np.testing.assert_array_equal(keys, load([tmp])[1])
            sel = np.array([0, 3, 14, 15, 16, 39])
            got, want = fields(packed, sel), fields(recs, sel)
            self.assertEqual(sorted(got), sorted(want))
            for name in want:
                np.testing.assert_array_equal(got[name], want[name], err_msg=name)
            for a, b in zip(batches(packed, sel, 4, "cpu"), batches(recs, sel, 4, "cpu")):
                for name in b:
                    torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)

    def test_empty_selection(self):
        self.assertEqual(list(batches(self.recs, np.array([], dtype=np.int64), 512, "cpu")), [])


class ContinuousTest(unittest.TestCase):
    def test_state_goes_on(self):
        """En continu, le deuxième pas reprend l'optimiseur du premier, et le réseau exporté est la
        moyenne des poids."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for c, seed in (("c001", 1), ("c002", 2)):
                (tmp / c).mkdir()
                random_records(64, np.random.default_rng(seed)).tofile(tmp / c / "a.pos")
                np.full(8, 8, "<u4").tofile(tmp / c / ("a.pos" + GAMES))
            torch.manual_seed(0)
            export(Net(16, 1), tmp / "c000.bin")

            def step(c, prev):
                argv = ["train.py", "--cycles", str(tmp), "--last", c, "--init", str(tmp / f"{prev}.bin"),
                        "--out", str(tmp / f"{c}.bin"), "--state-in", str(tmp / f"{prev}.state.pt"),
                        "--state-out", str(tmp / f"{c}.state.pt")]
                old, sys.argv = sys.argv, argv
                try:
                    with redirect_stdout(StringIO()) as out:
                        main()
                finally:
                    sys.argv = old
                return out.getvalue(), torch.load(tmp / f"{c}.state.pt", map_location="cpu")

            log1, s1 = step("c001", "c000")
            log2, s2 = step("c002", "c001")
            self.assertIn("premier pas", log1)
            self.assertIn("état repris", log2)
            # 20 lots par pas (le minimum de train.py)
            self.assertEqual(s1["opt"]["state"][0]["step"].item(), 20)
            self.assertEqual(s2["opt"]["state"][0]["step"].item(), 40)
            played = load_net(tmp / "c001.bin")
            for p, e in zip(played.parameters(), s1["ema"]):
                torch.testing.assert_close(p, e.cpu(), rtol=0, atol=0)
            raw = s1["net"]["node_in.weight"].cpu()
            self.assertFalse(torch.equal(played.node_in.weight, raw), "la moyenne n'est pas le réseau qui apprend")


if __name__ == "__main__":
    unittest.main()
