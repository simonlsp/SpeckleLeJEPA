import os
import struct
import tempfile
import unittest
import zlib
from collections import Counter
from contextlib import closing
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Event, Lock

import numpy as np
import torch

from speckle_lejepa import CellBatchStream, CellViews, CropRule, load_image_index
from train import training_batches


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

    def test_thread_counts_preserve_sampling_and_views(self) -> None:
        path, expected = self.make_index([
            ("A", (513, 65535, 42)), ("B", (600, 7)), ("C", (999,)), ("A", (1, 2)), ("D", ()),
        ])
        for cell_counts in ((0, 0), (1, 4)):
            with self.subTest(cell_counts=cell_counts):
                streams = [
                    CellBatchStream(path, batch_size=3, cells_per_image=2, cell_workers=c, image_workers=i)
                    for c, i in zip(cell_counts, (0, 2))
                ]
                reference, threaded = [list(stream) for stream in streams]
                self.assertEqual(streams[0].stats, streams[1].stats)
                self.assertEqual(streams[1].stats.empty_images, 1)
                self.assertEqual(streams[1].stats.emitted_cells, len(expected))
                self.assertEqual(len(reference), len(threaded))
                for a, b in zip(reference, threaded):
                    self.assertEqual(a.cells, b.cells)
                    torch.testing.assert_close(a.global_views, b.global_views, rtol=0, atol=0)
                    torch.testing.assert_close(a.local_views, b.local_views, rtol=0, atol=0)
                    self.assertLessEqual(max(Counter(cell.image_path for cell in b.cells).values()), 2)

    def test_next_batch_prepares_cells_in_parallel_before_consumer_advances(self) -> None:
        path, _ = self.make_index([("A", (1, 2)), ("B", (3, 4))])
        views = CellViews()
        lock = Lock()
        next_batch_workers_started = Event()
        finish_preparation = Event()
        calls = 0

        def controlled_views(*args):
            nonlocal calls
            with lock:
                calls += 1
                call_number = calls
                if call_number == 4:
                    next_batch_workers_started.set()
            if call_number > 2 and not finish_preparation.wait(5):
                raise RuntimeError("Timed out waiting for the test to release cell preparation")
            return views(*args)

        stream = CellBatchStream(path, batch_size=2, views=controlled_views, cell_workers=2, image_workers=2)
        with closing(training_batches(stream)) as batches:
            first, first_stats = next(batches)
            try:
                self.assertTrue(next_batch_workers_started.wait(5))
                self.assertEqual(first_stats.emitted_cells, 2)
            finally:
                finish_preparation.set()
            second, second_stats = next(batches)
            self.assertEqual(second_stats.emitted_cells, 4)
            self.assertEqual(first_stats.emitted_cells, 2)
            self.assertEqual(len(set(first.cells + second.cells)), 4)
            self.assertIsNone(next(batches, None))


if __name__ == "__main__":
    unittest.main()
