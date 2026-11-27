#!/usr/bin/env python3
"""Export one NumPy array of backbone embeddings per indexed gene."""

import argparse
import base64
import logging
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from speckle_lejepa import SpeckleLeJEPA
from speckle_lejepa.data import ImageRecord, load_cell_image, load_gene_index
from speckle_lejepa.views import masked_cell_crop
from train import positive_int

logger = logging.getLogger(__name__)


def gene_filename(gene: str) -> str:
    """Return the reversible filename for a UTF-8 gene name."""
    encoded = base64.b32encode(gene.encode("utf-8")).decode("ascii")
    return f"gene_{encoded}.npy"


def load_model(path: Path, device: torch.device) -> SpeckleLeJEPA:
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    backbone = {
        key.removeprefix("backbone."): tensor
        for key, tensor in state.items()
        if key.startswith("backbone.")
    }
    model = SpeckleLeJEPA()
    model.backbone.load_state_dict(backbone or state)
    return model.to(device).eval()


def cell_crops(
    records: Iterable[ImageRecord],
    max_cells: int | None,
    side: int,
    executor: ThreadPoolExecutor | None,
    prefetch: int,
) -> Iterator[torch.Tensor]:
    read = partial(load_cell_image, max_cell_per_image=max_cells)
    if executor is None:
        images = map(read, records)
    else:
        images = prefetch_images(records, read, executor, prefetch)
    for image in images:
        for instance in image.instances:
            pixels, mask = masked_cell_crop(
                image.rgb, image.labels, instance.label, instance.box, side
            )
            if 8 * mask.sum() >= mask.size:
                yield pixels


def prefetch_images(records, read, executor, depth):
    """Keep at most depth reads queued and yield them in index order."""
    remaining = iter(records)
    pending = deque(executor.submit(read, record) for record in islice(remaining, depth))
    while pending:
        image = pending.popleft().result()
        for record in islice(remaining, 1):
            pending.append(executor.submit(read, record))
        yield image


@torch.inference_mode()
def embed_cells(
    crops: Iterator[torch.Tensor], model: SpeckleLeJEPA, device: torch.device, batch_size: int
) -> np.ndarray:
    chunks = []
    while batch := list(islice(crops, batch_size)):
        features = model.encode(torch.stack(batch).to(device))
        chunks.append(features.cpu().numpy())
    if chunks:
        return np.concatenate(chunks)
    return np.empty((0, model.embedding_dim), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_path", type=Path, help="Path to a trusted ImageIndex NPY")
    parser.add_argument("model_path", type=Path, help="Training checkpoint or backbone state dict")
    parser.add_argument("--batch-size", "--batch_size", type=positive_int, default=256)
    parser.add_argument(
        "--max-image-per-gene", "--max_image_per_gene", type=positive_int,
        help="Take the first N indexed images per gene (default: all)",
    )
    parser.add_argument(
        "--max-cell-per-image", "--max_cell_per_image", type=positive_int,
        help="Check the first N instance IDs per image before applying the foreground threshold (default: all)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for gene_<base32>.npy files")
    parser.add_argument("--num-workers", type=int, default=2, help="Image prefetch threads (default: 2, 0 disables)")
    parser.add_argument("--device", help="Inference device (default: CUDA if available, otherwise CPU)")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")

    genes = load_gene_index(args.index_path)
    output_paths = {gene: args.output / gene_filename(gene) for gene in genes}
    input_paths = {args.index_path.resolve(), args.model_path.resolve()}
    for path in output_paths.values():
        if path.resolve() in input_paths:
            parser.error(f"Output path collision: {path}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.model_path, device)
    args.output.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=args.num_workers) if args.num_workers else nullcontext()
    with pool as executor, TemporaryDirectory(dir=args.output) as staging:
        for gene, images in genes.items():
            crops = cell_crops(
                islice(images, args.max_image_per_gene),
                args.max_cell_per_image, model.global_size, executor, args.num_workers,
            )
            embeddings = embed_cells(crops, model, device, args.batch_size)
            path = output_paths[gene]
            temporary = Path(staging) / path.name
            np.save(temporary, embeddings, allow_pickle=False)
            temporary.replace(path)
            logger.info("gene=%r cells=%d path=%s", gene, len(embeddings), path)
            del embeddings


if __name__ == "__main__":
    main()
