"""Régressions de la forme compacte écrite par le selfplay, du chargement des lots et de
l'apprentissage en continu. Les tests sur de vraies parties lancent `lab4 selfplay` (le programme
construit, `LAB4`, par défaut `target/release/lab4`) ; sautés s'il manque.

    cd lab4/py && .venv/bin/python -m unittest test_train
"""

import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch

from data import COMPACT, ILLEGAL, RECORD, Packed, games, load, map_keys, sources
from model import Net, export, load as load_net
from train import batches, main

ROOT = Path(os.environ.get("LAB4_ROOT", Path(__file__).resolve().parents[2]))
LAB4 = Path(os.environ.get("LAB4", ROOT / "target/release/lab4"))


def selfplay(out, seed, pos=False):
    """12 parties à 2, 3, 4 et 5 joueurs sur 2 fils (réseau de gen0, 16 simulations), dans `out`."""
    args = [LAB4, "selfplay", "--agent", f"az:net={ROOT / 'lab4/nets/gen0.bin'},n=16", "--games", "12", "--out", out,
            "--players", "2,3,4,5", "--seed", str(seed), "--threads", "2", "--device", "cpu", "--pos", str(pos).lower()]
    subprocess.run(args, check=True, capture_output=True)


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

    def test_empty_selection(self):
        self.assertEqual(list(batches(self.recs, np.array([], dtype=np.int64), 512, "cpu")), [])


@unittest.skipUnless(LAB4.exists(), f"{LAB4} absent")
class CompactTest(unittest.TestCase):
    def test_selfplay_writes_the_records(self):
        """La forme compacte redonne, au bit près, les enregistrements pleins des mêmes parties
        (`--pos true`), et la même clé de partie."""
        with tempfile.TemporaryDirectory() as tmp:
            selfplay(tmp, 7, pos=True)
            files = sources([tmp])
            self.assertEqual(len(files), 2)
            want = []
            for f in files:
                pos = f.with_name(f.name.removesuffix(COMPACT))
                records = np.fromfile(pos, dtype=RECORD)
                counts = games(pos, len(records))
                keys = np.repeat(map_keys(records[np.cumsum(counts) - counts]), counts)
                packed = Packed([f])
                self.assertGreater(len(packed), 20)
                np.testing.assert_array_equal(packed.keys, keys)
                got = packed.take(np.arange(len(packed)))
                for name in RECORD.names:
                    np.testing.assert_array_equal(got[name], records[name], err_msg=name)
                # la valeur de la recherche : une part de victoire par joueur présent
                present = np.arange(5)[None] < records["players"][:, None]
                np.testing.assert_allclose((got["value"] * present).sum(1), 1, rtol=1e-5)
                self.assertTrue((got["value"][~present] == 0).all())
                want.append(records)
            recs, _ = load([tmp])
            np.testing.assert_array_equal(recs, np.concatenate(want))


@unittest.skipUnless(LAB4.exists(), f"{LAB4} absent")
class TrainingTest(unittest.TestCase):
    def test_a_step_learns_and_exports(self):
        """Un pas d'apprentissage sur deux cycles joués : le réseau écrit se relit, et a appris."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for c, seed in (("c001", 1), ("c002", 2)):
                selfplay(tmp / c, seed)
            torch.manual_seed(0)
            export(Net(16, 1), tmp / "c000.bin")
            argv = ["train.py", "--cycles", str(tmp), "--last", "c002", "--init", str(tmp / "c000.bin"),
                    "--out", str(tmp / "c002.bin"), "--td", "0.5"]
            old, sys.argv = sys.argv, argv
            try:
                with redirect_stdout(StringIO()) as out:
                    main()
            finally:
                sys.argv = old
            self.assertIn("20 lots", out.getvalue())
            before, after = load_net(tmp / "c000.bin"), load_net(tmp / "c002.bin")
            self.assertFalse(torch.equal(before.node_in.weight, after.node_in.weight), "rien appris")

    def test_old_nets_load_without_the_owner_head(self):
        """c130, appris dans lab3 avec la tête des propriétaires, se lit sans elle."""
        net = load_net(ROOT / "lab4/nets/c130.bin")
        self.assertEqual((net.hidden, len(net.layers)), (128, 7))
        self.assertFalse(any(name.startswith("own.") for name in net.state_dict()))


if __name__ == "__main__":
    unittest.main()
