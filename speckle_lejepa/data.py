"""Read legacy microscopy indices and stream cell batches from a bounded image pool."""

import heapq
import os
from collections import defaultdict
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.io import decode_png

from .views import CellViews


@dataclass(frozen=True, slots=True)
class ImageRecord:
    gene: str
    rgb_path: PurePath
    mask_path: PurePath


class CellKey(NamedTuple):
    gene: str
    image_path: str
    label: int


class CellBatch(NamedTuple):
    global_views: Tensor
    local_views: Tensor
    cells: tuple[CellKey, ...]


@dataclass(slots=True)
class EpochStats:
    decoded_images: int = 0
    empty_images: int = 0
    emitted_cells: int = 0
    skipped_cells: int = 0
    peak_cached_images: int = 0


def load_image_index(index_path: str | Path) -> tuple[ImageRecord, ...]:
    """Read a trusted [gene-to-wells, id-to-wells, well-to-images] pickled NPY."""
    index_path = Path(index_path).absolute()
    _, records = _read_image_index(index_path)
    if not records:
        raise ValueError(f"No indexed RGB/mask pairs found in {index_path}.")
    return records


def load_gene_index(index_path: str | Path) -> dict[str, list[ImageRecord]]:
    """Group a trusted image index by gene, retaining genes without images."""
    genes, records = _read_image_index(Path(index_path).absolute())
    by_gene: dict[str, list[ImageRecord]] = {gene: [] for gene in genes}
    for record in records:
        by_gene[record.gene].append(record)
    return by_gene


def _read_image_index(index_path: Path) -> tuple[tuple[str, ...], tuple[ImageRecord, ...]]:
    index = np.load(index_path, allow_pickle=True)
    if not isinstance(index, np.ndarray) or index.shape != (3,):
        raise ValueError(f"{index_path} must contain the three legacy index dictionaries.")
    genes, gene_ids, wells = index
    if not all(isinstance(mapping, Mapping) for mapping in (genes, gene_ids, wells)):
        raise ValueError(f"{index_path} must contain the three legacy index dictionaries.")

    by_image: dict[PurePath, ImageRecord] = {}
    for gene, positions in genes.items():
        for position in positions:
            for row in wells.get(position, ()):
                if len(row) < 5:
                    raise ValueError(f"Image index row for gene {gene!r}, well {position!r} has fewer than five fields.")
                rgb_path = _index_path(index_path.parent, row[3])
                mask_path = _index_path(index_path.parent, row[4])
                mask_path = type(mask_path)(
                    *("SavedirPolished" if part == "Savedir" else part for part in mask_path.parts)
                )
                record = ImageRecord(str(gene), rgb_path, mask_path)
                previous = by_image.setdefault(rgb_path, record)
                if previous != record:
                    raise ValueError(f"Conflicting gene or mask entries for RGB image {rgb_path}.")
    return tuple(str(gene) for gene in genes), tuple(by_image.values())


def _index_path(directory: Path, value: str | os.PathLike) -> PurePath:
    path = Path(value)
    if not path.is_absolute():
        foreign_path = PurePosixPath(value) if os.name == "nt" else PureWindowsPath(value)
        if foreign_path.is_absolute():
            return foreign_path
    return Path(os.path.normpath(directory / path))


@dataclass(frozen=True, slots=True)
class _Instance:
    label: int
    box: tuple[int, int, int, int]


@dataclass(slots=True)
class _CachedImage:
    record: ImageRecord
    rgb: np.ndarray
    labels: np.ndarray
    instances: list[_Instance]


def _decode(path: PurePath) -> np.ndarray:
    if not isinstance(path, Path):
        raise OSError(f"Cannot read PNG {path}: absolute path uses a different platform's syntax.")
    try:
        content = bytearray(path.read_bytes())
        if len(content) <= 24 or content[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("expected a PNG file")
        decoded = decode_png(torch.frombuffer(content, dtype=torch.uint8))
        if content[24] == 16 and decoded.dtype != torch.uint16:
            raise ValueError("PNG decoder must preserve 16-bit samples")
        return decoded.numpy()
    except (OSError, RuntimeError, ValueError) as error:
        raise OSError(f"Cannot read PNG {path}: {error}") from error


def load_cell_image(
    record: ImageRecord, rng: np.random.Generator | None = None, max_cell_per_image: int | None = None
) -> _CachedImage:
    """Decode one image pair and collect instances in label order unless shuffled."""
    rgb = _decode(record.rgb_path)
    mask = _decode(record.mask_path)
    if rgb.shape[0] != 3 or rgb.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Expected a three-channel uint8 or uint16 RGB PNG at {record.rgb_path}.")
    if mask.shape[0] != 1 or mask.dtype != np.uint16:
        raise ValueError(f"Expected a single-channel 16-bit instance PNG at {record.mask_path}.")
    labels = mask[0]
    if rgb.shape[1:] != labels.shape:
        raise ValueError(f"RGB/mask image sizes differ for {record.rgb_path} and {record.mask_path}.")

    rows, columns = np.nonzero(labels)
    instances = []
    if rows.size:
        values = labels[rows, columns]
        entries = int(values.max()) + 1
        top = np.full(entries, labels.shape[0], dtype=np.int32)
        left = np.full(entries, labels.shape[1], dtype=np.int32)
        bottom = np.zeros(entries, dtype=np.int32)
        right = np.zeros(entries, dtype=np.int32)
        np.minimum.at(top, values, rows)
        np.minimum.at(left, values, columns)
        np.maximum.at(bottom, values, rows + 1)
        np.maximum.at(right, values, columns + 1)
        for label in np.flatnonzero(bottom):
            box = (int(top[label]), int(left[label]), int(bottom[label]), int(right[label]))
            instances.append(_Instance(int(label), box))
        if rng is not None:
            rng.shuffle(instances)
        if max_cell_per_image is not None:
            del instances[max_cell_per_image:]
    return _CachedImage(record, rgb, labels, instances)


class _ImagePool:
    def __init__(
        self,
        records: tuple[ImageRecord, ...],
        rng: np.random.Generator,
        stats: EpochStats,
        max_cell_per_image: int | None,
        readers: ThreadPoolExecutor | None,
        read_ahead: int,
    ) -> None:
        self.rng = rng
        self.stats = stats
        self.max_cell_per_image = max_cell_per_image
        self.readers = readers
        self.read_ahead = read_ahead
        self.reads: dict[ImageRecord, Future[_CachedImage]] = {}
        self.pending: dict[str, list[ImageRecord]] = defaultdict(list)
        self.active: dict[str, int] = defaultdict(int)
        self.generations: dict[str, int] = defaultdict(int)
        self.heap: list[tuple[int, float, str, int]] = []
        for record in records:
            self.pending[record.gene].append(record)
        for gene, images in self.pending.items():
            self.rng.shuffle(images)
            self._schedule(gene)

    def _schedule(self, gene: str) -> None:
        self.generations[gene] += 1
        if self.pending[gene]:
            heapq.heappush(
                self.heap, (self.active[gene], self.rng.random(), gene, self.generations[gene])
            )

    def acquire(self) -> _CachedImage | None:
        while self.heap:
            self.prefetch()
            if not self.heap:
                break
            _, _, gene, generation = heapq.heappop(self.heap)
            if generation != self.generations[gene]:
                continue
            record = self.pending[gene].pop()
            read = self.reads.pop(record, None)
            image = load_cell_image(record) if read is None else read.result()
            # Sampling stays on the coordinator and follows the original acquisition order.
            if image.instances:
                self.rng.shuffle(image.instances)
                if self.max_cell_per_image is not None:
                    del image.instances[self.max_cell_per_image:]
            self.stats.decoded_images += 1
            if image.instances:
                self.active[gene] += 1
                self._schedule(gene)
                return image
            self.stats.empty_images += 1
            del image
            self._schedule(gene)
        return None

    def prefetch(self) -> None:
        if self.readers is None or len(self.reads) >= self.read_ahead:
            return
        candidates = []
        while self.heap and len(candidates) < self.read_ahead:
            entry = heapq.heappop(self.heap)
            gene, generation = entry[2:]
            if generation == self.generations[gene]:
                candidates.append(entry)
        for entry in candidates:
            heapq.heappush(self.heap, entry)
        # Try one image per candidate gene first. With fewer genes, look farther into their queues.
        for records in zip_longest(*(reversed(self.pending[entry[2]]) for entry in candidates)):
            for record in records:
                if record is not None and record not in self.reads:
                    self.reads[record] = self.readers.submit(load_cell_image, record)
                    if len(self.reads) == self.read_ahead:
                        return

    def release(self, gene: str) -> None:
        self.active[gene] -= 1
        self._schedule(gene)


class CellBatchStream(IterableDataset):
    """Yield prebatched views from a bounded active pool and optional worker threads."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        batch_size: int = 256,
        cells_per_image: int = 1,
        max_image_per_gene: int | None = None,
        max_cell_per_image: int | None = None,
        views: CellViews | None = None,
        seed: int = 0,
        cell_workers: int = 0,
        image_workers: int = 0,
    ) -> None:
        super().__init__()
        if not 1 <= cells_per_image <= batch_size:
            raise ValueError("Require 1 <= cells_per_image <= batch_size.")
        if seed < 0:
            raise ValueError("Seed must be nonnegative.")
        if cell_workers < 0 or image_workers < 0:
            raise ValueError("Worker counts must be nonnegative.")
        if max_image_per_gene is not None and max_image_per_gene < 1:
            raise ValueError("max_image_per_gene must be positive.")
        if max_cell_per_image is not None and max_cell_per_image < 1:
            raise ValueError("max_cell_per_image must be positive.")
        self.records = load_image_index(index_path)
        if max_image_per_gene is not None:
            rng = np.random.default_rng(seed)
            by_gene: dict[str, list[ImageRecord]] = defaultdict(list)
            for record in self.records:
                by_gene[record.gene].append(record)
            for images in by_gene.values():
                if len(images) > max_image_per_gene:
                    rng.shuffle(images)
                    del images[max_image_per_gene:]
            self.records = tuple(record for images in by_gene.values() for record in images)
        self.batch_size = batch_size
        self.cells_per_image = cells_per_image
        self.max_cell_per_image = max_cell_per_image
        self.cache_capacity = (batch_size + cells_per_image - 1) // cells_per_image
        self.views = views if views is not None else CellViews()
        self.seed = seed
        self.cell_workers = cell_workers
        self.image_workers = image_workers
        self.epoch = 0
        self.stats = EpochStats()

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative.")
        self.epoch = epoch

    def __iter__(self) -> Iterator[CellBatch]:
        if get_worker_info() is not None:
            raise RuntimeError("Use num_workers=0 so this stream owns a single image pool.")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        self.stats = EpochStats()
        readers = ThreadPoolExecutor(self.image_workers, thread_name_prefix="image-read") if self.image_workers else None
        workers = ThreadPoolExecutor(self.cell_workers, thread_name_prefix="cell-view") if self.cell_workers else None
        try:
            pool = _ImagePool(self.records, rng, self.stats, self.max_cell_per_image, readers, self.image_workers)
            crop_seeds = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 1]))
            yield from self._batches(pool, rng, workers, crop_seeds)
        finally:
            for executor in (workers, readers):
                if executor is not None:
                    executor.shutdown(cancel_futures=True)

    def _prepare(
        self, image: _CachedImage, limit: int, rng: np.random.Generator
    ) -> tuple[list[Tensor], list[Tensor], list[CellKey], int]:
        global_views, local_views, cells = [], [], []
        skipped = 0
        for _ in range(limit):
            while image.instances:
                instance = image.instances.pop()
                views = self.views(image.rgb, image.labels, instance.label, instance.box, rng)
                if views is not None:
                    global_views.append(views[0])
                    local_views.append(views[1])
                    cells.append(CellKey(image.record.gene, str(image.record.rgb_path), instance.label))
                    break
                skipped += 1
            else:
                break
        return global_views, local_views, cells, skipped

    def _batches(
        self, pool: _ImagePool, rng: np.random.Generator,
        workers: ThreadPoolExecutor | None, crop_seeds: np.random.Generator,
    ) -> Iterator[CellBatch]:
        active: list[_CachedImage] = []
        while True:
            while len(active) < self.cache_capacity:
                image = pool.acquire()
                if image is None:
                    break
                active.append(image)
                del image
            if not active:
                break
            self.stats.peak_cached_images = max(self.stats.peak_cached_images, len(active))
            rng.shuffle(active)

            global_views, local_views, cells = [], [], []
            exhausted = []
            visiting, active = active, []

            def retain(image: _CachedImage) -> None:
                if image.instances:
                    active.append(image)
                else:
                    exhausted.append(image.record.gene)
                    pool.prefetch()

            def collect(prepared: tuple[list[Tensor], list[Tensor], list[CellKey], int]) -> None:
                image_global, image_local, image_cells, skipped = prepared
                global_views.extend(image_global)
                local_views.extend(image_local)
                cells.extend(image_cells)
                self.stats.skipped_cells += skipped

            def crop_rng() -> np.random.Generator:
                if workers is None:
                    return rng
                return np.random.default_rng(int(crop_seeds.integers(2**63)))

            # Each worker owns one image at a time. Reserve only as many cells as the batch
            # can accept, then collect in sampling order even when workers finish out of order.
            while visiting:
                remaining = self.batch_size - len(cells)
                if remaining == 0:
                    retain(visiting.pop())
                elif workers is None:
                    image = visiting.pop()
                    collect(self._prepare(image, min(remaining, self.cells_per_image), rng))
                    retain(image)
                    del image
                else:
                    jobs = []
                    while visiting and remaining:
                        image = visiting.pop()
                        limit = min(remaining, self.cells_per_image)
                        image_rng = crop_rng()
                        jobs.append((image, limit, image_rng, workers.submit(self._prepare, image, limit, image_rng)))
                        remaining -= limit
                    for image, limit, image_rng, job in jobs:
                        collect(job.result())
                        # A short result from an earlier image can free space for the last job.
                        extra = min(self.batch_size - len(cells), self.cells_per_image - limit)
                        if extra and image.instances:
                            collect(self._prepare(image, extra, image_rng))
                        retain(image)
                    jobs.clear()
                    del image, job
            while len(cells) < self.batch_size and len(active) < self.cache_capacity:
                image = pool.acquire()
                if image is None:
                    break
                self.stats.peak_cached_images = max(self.stats.peak_cached_images, len(active) + 1)
                limit = min(self.batch_size - len(cells), self.cells_per_image)
                collect(self._prepare(image, limit, crop_rng()))
                retain(image)
                del image
            for gene in exhausted:
                pool.release(gene)
            pool.prefetch()
            if cells:
                self.stats.emitted_cells += len(cells)
                yield CellBatch(torch.stack(global_views), torch.stack(local_views), tuple(cells))
