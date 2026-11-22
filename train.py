#!/usr/bin/env python3
"""Train SpeckleLeJEPA from a legacy ImageIndex NPY and save epoch checkpoints."""

import argparse
import logging
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from speckle_lejepa import CellBatchStream, MixedPrecision, SpeckleLeJEPA

logger = logging.getLogger(__name__)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected an integer greater than zero")
    return number


def find_checkpoints(directory: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in directory.glob("epoch_*.pt"):
        match = re.fullmatch(r"epoch_([0-9]+)\.pt", path.name)
        if match and path.is_file():
            checkpoints.append((int(match[1]), path))
    return sorted(checkpoints)


def save_checkpoint(directory: Path, state: dict, max_saved_checkpoints: int | None) -> None:
    path = directory / f"epoch_{state['epoch']:04d}.pt"
    with TemporaryDirectory(dir=directory) as temp:
        pending = Path(temp) / path.name
        torch.save(state, pending)
        pending.replace(path)
    logger.info("checkpoint_saved epoch=%d path=%s", state["epoch"], path)
    if max_saved_checkpoints is not None:
        for epoch, old_path in find_checkpoints(directory)[:-max_saved_checkpoints]:
            old_path.unlink(missing_ok=True)
            logger.info("checkpoint_removed epoch=%d path=%s", epoch, old_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_path", type=Path, help="Path to a trusted ImageIndex NPY")
    parser.add_argument(
        "--batch-size", "--batch_size", type=positive_int, default=256, help="Maximum cells per batch (default: 256)"
    )
    parser.add_argument(
        "--max-image-per-gene", "--max_image_per_gene", type=positive_int, help="Image limit per gene (default: all)"
    )
    parser.add_argument(
        "--max-cell-per-image", "--max_cell_per_image", type=positive_int, help="Instance limit per image per epoch (default: all)"
    )
    parser.add_argument("--epochs", type=positive_int, help="Final epoch number. Defaults to running one epoch.")
    parser.add_argument("--lr", type=float, help="AdamW learning rate (default for new runs: 5e-4)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="Training device. Defaults to CUDA when available, then CPU.")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"), help="Checkpoint and log directory")
    parser.add_argument(
        "--max-saved-checkpoints", "--max_saved_checkpoints", type=positive_int,
        help="Number of newest epoch checkpoints to retain (default: all)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in --output")
    parser.add_argument(
        "--log-interval", "--log_interval", type=positive_int, default=100, help="Log every N batches (default: 100)"
    )
    args = parser.parse_args()
    if args.output.exists() and not args.output.is_dir():
        parser.error(f"--output must be a directory: {args.output}")
    checkpoints = find_checkpoints(args.output)
    if args.resume and not checkpoints:
        parser.error(f"--resume requires an epoch checkpoint in {args.output}")
    if not args.resume and checkpoints:
        parser.error(f"{args.output} already contains epoch checkpoints. Use --resume or a different --output.")

    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.output / "train.log", encoding="utf-8")],
    )

    completed_epochs, global_step = 0, 0
    resume_path = None
    if args.resume:
        completed_epochs, resume_path = checkpoints[-1]
    final_epoch = args.epochs if args.epochs is not None else completed_epochs + 1
    if final_epoch <= completed_epochs:
        logger.info("training_complete completed_epoch=%d target_epoch=%d", completed_epochs, final_epoch)
        return

    logger.info("index_loading path=%s", args.index_path)
    stream = CellBatchStream(
        args.index_path,
        batch_size=args.batch_size,
        max_image_per_gene=args.max_image_per_gene,
        max_cell_per_image=args.max_cell_per_image,
        seed=args.seed,
    )
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SpeckleLeJEPA().to(device).train()
    precision = MixedPrecision.for_device(device)
    scaler = precision.grad_scaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4 if args.lr is None else args.lr)
    if resume_path is not None:
        logger.info("checkpoint_loading epoch=%d path=%s", completed_epochs, resume_path)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if checkpoint["epoch"] != completed_epochs:
            raise ValueError(f"Checkpoint epoch does not match its filename: {resume_path}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        if scaler.is_enabled() and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["cpu_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        global_step = checkpoint["global_step"]
        del checkpoint
        logger.info("checkpoint_restored epoch=%d global_step=%d", completed_epochs, global_step)

    logger.info(
        "training_started index=%s output=%s epochs=%d..%d images=%d batch_size=%d device=%s precision=%s "
        "lr=%g seed=%d max_image_per_gene=%s max_cell_per_image=%s max_saved_checkpoints=%s",
        args.index_path, args.output, completed_epochs + 1, final_epoch, len(stream.records), stream.batch_size,
        device, precision.dtype or torch.float32, optimizer.param_groups[0]["lr"], args.seed,
        args.max_image_per_gene, args.max_cell_per_image, args.max_saved_checkpoints,
    )

    for epoch in range(completed_epochs + 1, final_epoch + 1):
        logger.info("epoch_started epoch=%d target_epoch=%d", epoch, final_epoch)
        started = perf_counter()
        stream.set_epoch(epoch - 1)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        totals = torch.zeros(3, device=device, dtype=torch.float32)
        steps = 0
        for steps, batch in enumerate(stream, start=1):
            global_views = batch.global_views.to(device)
            local_views = batch.local_views.to(device)
            optimizer.zero_grad(set_to_none=True)
            with precision.autocast():
                losses = model(global_views, local_views)
            scaler.scale(losses.total).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            values = torch.stack(tuple(loss.detach() for loss in losses))
            totals.add_(values, alpha=len(batch.cells))
            if steps == 1 or steps % args.log_interval == 0:
                total, prediction, sigreg = values.tolist()
                elapsed = perf_counter() - started
                stats = stream.stats
                logger.info(
                    "batch epoch=%d step=%d global_step=%d batch_size=%d loss=%.6f prediction=%.6f sigreg=%.6f "
                    "images=%d/%d cells=%d skipped_cells=%d elapsed_s=%.2f cells_per_s=%.2f",
                    epoch, steps, global_step, len(batch.cells), total, prediction, sigreg,
                    stats.decoded_images, len(stream.records), stats.emitted_cells, stats.skipped_cells,
                    elapsed, stats.emitted_cells / elapsed,
                )

        stats = stream.stats
        if steps == 0:
            raise RuntimeError(f"No trainable cells in epoch {epoch}. {stats}")
        if not torch.isfinite(totals).all():
            raise RuntimeError(f"Non-finite training loss in epoch {epoch}.")
        total, prediction, sigreg = (totals / stats.emitted_cells).tolist()
        elapsed = perf_counter() - started
        peak_cuda_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0
        logger.info(
            "epoch_finished epoch=%d steps=%d global_step=%d cells=%d images=%d empty_images=%d skipped_cells=%d "
            "peak_cached_images=%d mean_loss=%.6f mean_prediction=%.6f mean_sigreg=%.6f "
            "elapsed_s=%.2f cells_per_s=%.2f peak_cuda_mib=%.1f",
            epoch, steps, global_step, stats.emitted_cells, stats.decoded_images, stats.empty_images,
            stats.skipped_cells, stats.peak_cached_images, total, prediction, sigreg,
            elapsed, stats.emitted_cells / elapsed, peak_cuda_mib,
        )

        save_checkpoint(
            args.output,
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            },
            args.max_saved_checkpoints,
        )
    logger.info("training_complete completed_epoch=%d global_step=%d", final_epoch, global_step)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("training_failed")
        raise SystemExit(1)
