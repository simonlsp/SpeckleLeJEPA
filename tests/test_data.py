import os
import struct
import tempfile
import unittest
import zlib
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import torch

from speckle_lejepa import CellBatchStream, CellViews, CropRule, load_image_index


def write_png(path: Path, array: np.ndarray) -> None:
    def chunk(kind: bytes, content: bytes) -> bytes:
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content))

    height, width = array.shape[:2]
    bit_depth = 8 * array.dtype.itemsize
    color = 0 if array.ndim == 2 else 2
    header = struct.pack(">IIBBBBB", width, height, bit_depth, color, 0, 0, 0)
    rows = array.astype(array.dtype.newbyteorder(">"), copy=False)
    pixels = zlib.compress(b"".join(b"\x00" + row.tobytes() for row in rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b""))


class CellBatchStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, cls.previous_threads))

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)

    def make_index(self, specs: list[tuple[str, tuple[int, ...]]]) -> tuple[Path, set[tuple[str, int]]]:
        genes: dict[str, list[tuple[int, int, int]]] = {}
        wells = {}
        cells = set()
        for index, (gene, labels) in enumerate(specs):
            position = (1, 1, index)
            genes.setdefault(gene, []).append(position)
            rgb_path = Path("图像") / f"{index}_cha.png"
            mask_path = Path("Savedir") / f"{index}_seg.png"
            wells[position] = [("ch1", "ch2", "ch3", str(rgb_path), str(mask_path))]
            rgb = np.full((96, 96, 3), 65535, dtype=np.uint16)
            mask = np.zeros((96, 96), dtype=np.uint16)
            for elt, label in enumerate(labels):
                y, x = divmod(elt, 2)
                region = (slice(y * 40, y * 40 + 25), slice(x * 40, x * 40 + 30))
                mask[region] = label
                rgb[region] = [4096, 16000, 50000]
                cells.add((str(self.directory / rgb_path), label))
            write_png(self.directory / rgb_path, rgb)
            write_png(self.directory / "SavedirPolished" / mask_path.name, mask)
        genes["unindexed"] = [(1, 1, 99)]
        path = self.directory / "index.npy"
        np.save(path, np.array([genes, {}, wells], dtype=object))
        return path, cells

    def test_paths_uint16_labels_and_exact_epoch_coverage(self) -> None:
        path, expected = self.make_index([
            ("A", (513, 65535, 42)), ("B", (600, 7)), ("C", (999,)),
            ("A", (1, 2)), ("A", (1024,)), ("B", ()),
        ])
        self.assertEqual(len(load_image_index(path)), 6)
        stream = CellBatchStream(path, batch_size=3)
        batches = list(stream)
        self.assertEqual({cell.gene for cell in batches[0].cells}, {"A", "B", "C"})
        self.assertEqual([len(batch.cells) for batch in batches[:2]], [3, 3])
        self.assertTrue(all(0 < len(batch.cells) <= 3 for batch in batches))
        seen = [(cell.image_path, cell.label) for batch in batches for cell in batch.cells]
        self.assertEqual(set(seen), expected)
        self.assertEqual(len(seen), len(expected))
        self.assertEqual(stream.stats.decoded_images, 6)
        self.assertEqual(stream.stats.empty_images, 1)
        self.assertEqual(stream.stats.peak_cached_images, 3)
        for batch in batches:
            self.assertEqual(len({cell.image_path for cell in batch.cells}), len(batch.cells))
            self.assertEqual(batch.global_views.shape[1:], (2, 3, 96, 96))
            self.assertEqual(batch.local_views.shape[1:], (2, 3, 48, 48))
            for tensor, minimum in ((batch.global_views, 0.15), (batch.local_views, 0.45)):
                self.assertTrue(((tensor[:, :, 0] > 0).float().mean(dim=(-1, -2)) >= minimum).all())
                channel_max = torch.tensor([4096, 16000, 50000]) / 65535
                self.assertTrue((tensor.amax(dim=(0, 1, 3, 4)) <= channel_max + 1e-6).all())

    def test_multiple_cells_per_image_backfill_and_tail(self) -> None:
        path, expected = self.make_index([
            ("A", (330,)), ("A", (7, 8)), ("B", (1, 2)), ("A", (4, 5)),
        ])
        stream = CellBatchStream(path, batch_size=4, cells_per_image=2)
        batches = list(stream)
        self.assertEqual([len(batch.cells) for batch in batches], [4, 3])
        self.assertEqual(stream.stats.peak_cached_images, 2)
        for batch in batches:
            self.assertLessEqual(max(Counter(cell.image_path for cell in batch.cells).values()), 2)
        seen = [(cell.image_path, cell.label) for batch in batches for cell in batch.cells]
        self.assertEqual(len(seen), len(expected))
        self.assertEqual(set(seen), expected)

    def test_index_preserves_foreign_absolute_paths(self) -> None:
        path_type = PurePosixPath if os.name == "nt" else PureWindowsPath
        directory = path_type("/remote/images" if os.name == "nt" else "Z:/remote/images")
        rgb = directory / "rgb.png"
        mask = directory / "Savedir" / "mask.png"
        position = (1, 1, 1)
        path = self.directory / "index.npy"
        wells = {position: [("", "", "", str(rgb), str(mask))]}
        np.save(path, np.array([{"A": [position]}, {}, wells], dtype=object))

        record, = load_image_index(path)
        self.assertIs(type(record.rgb_path), path_type)
        self.assertEqual(record.rgb_path, rgb)
        self.assertEqual(record.mask_path, directory / "SavedirPolished" / "mask.png")

    def test_sampling_limits_apply_independently(self) -> None:
        path, _ = self.make_index([("A", (1, 2, 3)), ("A", (4, 5, 6)), ("B", (7, 8, 9))])
        cases = [
            ({"max_image_per_gene": 1}, 2, 3),
            ({"max_cell_per_image": 2}, 3, 2),
            ({"max_image_per_gene": 1, "max_cell_per_image": 2}, 2, 2),
        ]
        for limits, image_count, cell_count in cases:
            with self.subTest(limits=limits):
                stream = CellBatchStream(path, batch_size=3, seed=7, **limits)
                for epoch in range(2):
                    stream.set_epoch(epoch)
                    cells = [cell for batch in stream for cell in batch.cells]
                    per_image = Counter(cell.image_path for cell in cells)
                    self.assertEqual({cell.gene for cell in cells}, {"A", "B"})
                    self.assertEqual(len(per_image), image_count)
                    self.assertEqual(set(per_image.values()), {cell_count})
                    self.assertEqual(len(set(cells)), len(cells))
                    self.assertEqual(stream.stats.decoded_images, image_count)

    def test_impossible_crop_is_counted(self) -> None:
        path, _ = self.make_index([("A", (512,))])
        views = CellViews(global_crop=CropRule(2, 2, 1, 1))
        stream = CellBatchStream(path, batch_size=1, views=views)
        self.assertEqual(list(stream), [])
        self.assertEqual(stream.stats.skipped_cells, 1)


if __name__ == "__main__":
    unittest.main()
