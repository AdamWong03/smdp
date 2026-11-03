"""Sampling-aware thermodynamic, spectral and structural statistics."""

from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from periodic import PeriodicGraphBuilder
from physics.metrics import reference_vdos


def _spectrum_worker_init(filename, dataset_name, numbers, start, stride,
                          block_frames, dt):
    import h5py
    global _spectral_input
    handle = h5py.File(filename, "r")
    _spectral_input = (
        handle, handle[dataset_name], numbers, start, stride, block_frames, dt
    )


def _spectrum_worker(index):
    _, dataset, numbers, start, stride, block_frames, dt = _spectral_input
    offset = index * block_frames
    values = np.asarray(
        dataset[
            start + offset * stride:
            start + (offset + block_frames) * stride:
            stride
        ],
        dtype=np.float64,
    )
    record = {}
    for atomic_number in np.unique(numbers):
        frequency, power = reference_vdos(
            values[:, numbers == atomic_number], dt
        )
        record[str(atomic_number)] = power
    return index, frequency, record


def stats(values):
    values = np.asarray(values, dtype=float)
    blocks = [
        block.mean()
        for block in np.array_split(values, min(8, len(values)))
        if len(block)
    ]
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "variance": float(values.var()),
        "block_mean_std": float(np.std(blocks)),
        "blocks": len(blocks),
    }


def paired(candidate, reference):
    return {
        "SMDP": stats(candidate),
        "reference": stats(reference),
        "mean_difference": float(np.mean(candidate) - np.mean(reference)),
        "wasserstein": float(wasserstein_distance(candidate, reference)),
    }


def spectrum_blocks(dataset, numbers, start, frames, stride, dt, cache=None,
                    workers=1):
    """Compute non-overlapping 16.384 ps VDOS blocks with safe resume."""
    block_frames = min(int(16384 / dt), frames)
    if block_frames * dt < 4096:
        return None
    accum = {int(z): [] for z in np.unique(numbers)}
    offsets = range(0, frames - block_frames + 1, block_frames)
    if cache is not None:
        Path(cache).mkdir(parents=True, exist_ok=True)
    if workers > 1:
        if cache is None:
            raise ValueError("Parallel spectral analysis requires resumable cache")
        missing = [
            i for i in range(len(offsets))
            if not (Path(cache) / f"{i:06d}.npz").exists()
        ]
        if missing:
            from multiprocessing import get_context
            with get_context("spawn").Pool(
                workers,
                initializer=_spectrum_worker_init,
                initargs=(dataset.file.filename, dataset.name, numbers, start,
                          stride, block_frames, dt),
            ) as pool:
                for completed, (index, frequency, record) in enumerate(
                    pool.imap_unordered(_spectrum_worker, missing), 1
                ):
                    path = Path(cache) / f"{index:06d}.npz"
                    temporary = path.with_suffix(".tmp")
                    with temporary.open("wb") as handle:
                        np.savez(handle, frequency=frequency, **record)
                    temporary.replace(path)
                    if completed % 64 == 0 or completed == len(missing):
                        done = len(offsets) - len(missing) + completed
                        print(
                            f"VDOS {Path(cache).name}: {done}/{len(offsets)} blocks",
                            flush=True,
                        )
    for index, offset in enumerate(offsets):
        path = Path(cache) / f"{index:06d}.npz" if cache is not None else None
        if path is not None and path.exists():
            with np.load(path) as saved:
                frequency = saved["frequency"]
                for atomic_number in accum:
                    accum[atomic_number].append(saved[str(atomic_number)])
            continue
        values = np.asarray(
            dataset[
                start + offset * stride:
                start + (offset + block_frames) * stride:
                stride
            ],
            dtype=np.float64,
        )
        record = {}
        for atomic_number in accum:
            frequency, power = reference_vdos(
                values[:, numbers == atomic_number], dt
            )
            accum[atomic_number].append(power)
            record[str(atomic_number)] = power
        if path is not None:
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, frequency=frequency, **record)
            temporary.replace(path)
    return frequency, {z: np.asarray(values) for z, values in accum.items()}


def rdf_sample(frames, cell, numbers, first, second, rmax, bins=180):
    builder = PeriodicGraphBuilder(rmax)
    edges = np.linspace(0, rmax, bins + 1)
    histograms = []
    for positions in frames:
        edge_index, shifts = builder.one(positions, cell)
        mask = (
            (numbers[edge_index[1]] == first)
            & (numbers[edge_index[0]] == second)
        )
        distances = np.linalg.norm(
            positions[edge_index[0, mask]]
            + shifts[mask]
            - positions[edge_index[1, mask]],
            axis=1,
        )
        histograms.append(np.histogram(distances, edges)[0])
    shell = 4 * np.pi / 3 * np.diff(edges ** 3)
    volume = abs(np.linalg.det(cell))
    denominator = (
        np.sum(numbers == first)
        * (np.sum(numbers == second) - (first == second))
        / volume
        * shell
    )
    radii = 0.5 * (edges[:-1] + edges[1:])
    return radii, np.asarray(histograms) / np.maximum(denominator, 1e-30)
