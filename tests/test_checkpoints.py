import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from train import find_checkpoints, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)

    def test_retention_uses_numeric_epoch_and_preserves_other_files(self) -> None:
        for name in ("epoch_9.pt", "epoch_10.pt", "backup_1.pt"):
            (self.directory / name).touch()

        save_checkpoint(self.directory, {"epoch": 11}, None)
        self.assertEqual([epoch for epoch, _ in find_checkpoints(self.directory)], [9, 10, 11])
        save_checkpoint(self.directory, {"epoch": 12}, 2)

        checkpoints = find_checkpoints(self.directory)
        self.assertEqual([epoch for epoch, _ in checkpoints], [11, 12])
        self.assertEqual(torch.load(checkpoints[-1][1], weights_only=True), {"epoch": 12})
        self.assertTrue((self.directory / "backup_1.pt").is_file())

    def test_failed_save_keeps_previous_checkpoint(self) -> None:
        save_checkpoint(self.directory, {"epoch": 1}, None)
        with patch("train.torch.save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                save_checkpoint(self.directory, {"epoch": 2}, 1)

        self.assertEqual(sorted(path.name for path in self.directory.iterdir()), ["epoch_0001.pt"])
