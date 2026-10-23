"""Régressions du chargement des lots, sans apprentissage ni export de réseau.

    cd lab3/py && .venv/bin/python -m unittest test_train
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data import ILLEGAL, RECORD, MappedRecords
from train import batches


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

    def test_final_single_position_in_ram_and_mmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Le dernier lot reprend une position du deuxième fichier.
            paths = [Path(tmp) / "a.pos", Path(tmp) / "b.pos"]
            self.recs[:1].tofile(paths[0])
            self.recs[1:].tofile(paths[1])
            mapped = MappedRecords([(paths[0], 1), (paths[1], 2)])
            for recs in (self.recs, mapped):
                with self.subTest(mapped=isinstance(recs, MappedRecords)):
                    bs = list(batches(recs, np.array([2, 0, 1]), 2, "cpu"))
                    self.assertEqual([len(b["win"]) for b in bs], [2, 1])
                    # Les indices sont triés dans chaque lot, pas dans toute la séquence.
                    torch.testing.assert_close(bs[0]["win"], torch.from_numpy(self.recs["win"][[0, 2]].copy()))
                    torch.testing.assert_close(bs[1]["win"], torch.from_numpy(self.recs["win"][[1]].copy()))

    def test_empty_selection(self):
        self.assertEqual(list(batches(self.recs, np.array([], dtype=np.int64), 512, "cpu")), [])


if __name__ == "__main__":
    unittest.main()
