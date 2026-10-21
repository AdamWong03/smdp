"""Physical trajectory metrics used by the quartz benchmark."""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window
from scipy.stats import wasserstein_distance

from .units import KB_EV_PER_K, KINETIC_TO_EV


def kinetic_energy_ev(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return 0.5 * KINETIC_TO_EV * np.sum(
        masses.reshape(1, -1, 1) * np.asarray(velocities) ** 2, axis=(1, 2)
    )


def temperature_k(
    velocities: np.ndarray, masses: np.ndarray, *, removed_dof: int = 3
) -> np.ndarray:
    degrees = 3 * len(masses) - removed_dof
    return 2.0 * kinetic_energy_ev(velocities, masses) / (degrees * KB_EV_PER_K)


def element_temperatures_k(
    velocities: np.ndarray, masses: np.ndarray, atomic_numbers: np.ndarray
) -> dict[int, np.ndarray]:
    result = {}
    for atomic_number in np.unique(atomic_numbers):
        mask = atomic_numbers == atomic_number
        kinetic = 0.5 * KINETIC_TO_EV * np.sum(
            masses[mask].reshape(1, -1, 1) * velocities[:, mask] ** 2,
            axis=(1, 2),
        )
        result[int(atomic_number)] = 2.0 * kinetic / (
            3 * int(mask.sum()) * KB_EV_PER_K
        )
    return result


def velocity_autocorrelation_fft(velocities: np.ndarray) -> np.ndarray:
    """Unbiased atom/component-averaged VACF using FFT convolution."""
    values = np.asarray(velocities, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    n = values.shape[0]
    transform = np.fft.rfft(values, n=2 * n, axis=0)
    correlation = np.fft.irfft(transform * transform.conj(), n=2 * n, axis=0)[:n]
    correlation /= np.arange(n, 0, -1).reshape(-1, 1, 1)
    vacf = correlation.mean(axis=(1, 2))
    return vacf / max(vacf[0], np.finfo(np.float64).tiny)


def vdos(
    velocities: np.ndarray,
    dt_fs: float,
    *,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    vacf = velocity_autocorrelation_fft(velocities)
    weights = get_window(window, len(vacf), fftbins=False)
    spectrum = np.real(np.fft.rfft(vacf * weights))
    spectrum = np.maximum(spectrum, 0.0)
    frequency_thz = np.fft.rfftfreq(len(vacf), dt_fs * 1.0e-15) / 1.0e12
    area = np.trapezoid(spectrum, frequency_thz)
    if area > 0.0:
        spectrum /= area
    return frequency_thz, spectrum


def reference_vdos(
    velocities: np.ndarray,
    dt_fs: float,
    *,
    correlation_window_fs: float = 2000.0,
    padding_window_fs: float = 2000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """VDOS used by the validation method cited by TrajCast.

    This reproduces ``MarsalekGroup/aml:aml/score/vdos.py`` without forming
    the full ``2*T-1`` autocorrelation.  AML keeps a centered 2 ps slice of
    the unbiased, *uncentered* velocity autocorrelation, applies a symmetric
    Hann window, zero-pads by another 2 ps, and takes the absolute rFFT.  The
    returned intensity is normalized by a discrete sum, as in AML.

    ``velocities`` is one species with shape ``(time, atoms, 3)``.  Only the
    lags retained by the 2 ps window are evaluated, so memory is O(T*N) and
    work is O(T*N*window/dt), rather than O(T**2).
    """
    values = np.asarray(velocities)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("velocities must have shape (time, atoms, 3)")
    n_frames = values.shape[0]
    n_atoms = values.shape[1]
    n_window = int(correlation_window_fs / dt_fs / 2.0)
    n_padding = int(padding_window_fs / dt_fs / 2.0)
    if 2 * n_window + 1 > 2 * n_frames - 1:
        raise ValueError("correlation window is wider than the trajectory")

    # AML averages atom-resolved dot-product autocorrelations.  Summing the
    # component power spectra before a single inverse FFT is algebraically
    # identical to summing those correlations, while avoiding O(T*window*N)
    # work for the 1 fs high-resolution validation trajectories.
    flattened = np.asarray(values, dtype=np.float64).reshape(n_frames, -1)
    n_fft = 1 << (2 * n_frames - 2).bit_length()
    transform = np.fft.rfft(flattened, n=n_fft, axis=0)
    summed_power = np.einsum(
        "fd,fd->f", transform.conj(), transform, dtype=np.complex128, optimize=True
    )
    raw_positive = np.fft.irfft(summed_power, n=n_fft)[: n_window + 1]
    positive = raw_positive / (
        (n_frames - np.arange(n_window + 1, dtype=np.float64)) * n_atoms
    )
    correlation = np.concatenate((positive[:0:-1], positive))
    correlation *= np.hanning(2 * n_window + 1)
    correlation = np.pad(correlation, (n_padding + 1, n_padding))
    intensity = np.abs(np.fft.rfft(correlation))
    intensity /= intensity.sum()
    frequency_thz = np.fft.rfftfreq(len(correlation), dt_fs * 1.0e-3)
    return frequency_thz, intensity


def spectral_overlap(
    frequency: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    reference = np.maximum(np.asarray(reference, dtype=np.float64), 0.0)
    candidate = np.maximum(np.asarray(candidate, dtype=np.float64), 0.0)
    reference /= np.trapezoid(reference, frequency)
    candidate /= np.trapezoid(candidate, frequency)
    return float(np.trapezoid(np.minimum(reference, candidate), frequency))


def minimum_image_displacements(
    positions: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    difference = positions[:, None, :] - positions[None, :, :]
    from ase.geometry import find_mic
    # Component-wise fractional rounding is not general for skew cells.
    mapped, _ = find_mic(difference.reshape(-1, 3), cell, pbc=True)
    return mapped.reshape(difference.shape)


def rdf(
    positions: np.ndarray,
    cell: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    *,
    r_max: float,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    histogram = np.zeros(bins, dtype=np.float64)
    edges = np.linspace(0.0, r_max, bins + 1)
    same = np.array_equal(first_mask, second_mask)
    for frame in np.asarray(positions):
        delta = minimum_image_displacements(frame, cell)
        selected = np.linalg.norm(delta[first_mask][:, second_mask], axis=-1)
        if same:
            selected = selected[np.triu_indices(selected.shape[0], k=1)]
            pair_factor = 2.0
        else:
            selected = selected.reshape(-1)
            pair_factor = 1.0
        histogram += np.histogram(selected, bins=edges)[0] * pair_factor
    centers = 0.5 * (edges[1:] + edges[:-1])
    shell_volume = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = abs(np.linalg.det(cell))
    n_first = int(first_mask.sum())
    n_second = int(second_mask.sum())
    density_second = (n_second - (1 if same else 0)) / volume
    normalization = len(positions) * n_first * density_second * shell_volume
    return centers, histogram / np.maximum(normalization, np.finfo(float).tiny)


def minimum_pair_distance(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    result = np.empty(len(positions), dtype=np.float64)
    for index, frame in enumerate(positions):
        distances = np.linalg.norm(minimum_image_displacements(frame, cell), axis=-1)
        np.fill_diagonal(distances, np.inf)
        result[index] = distances.min()
    return result


def linear_energy_drift_mev_per_atom_ps(
    energy_ev: np.ndarray, dt_fs: float, n_atoms: int
) -> float:
    time_ps = np.arange(len(energy_ev), dtype=np.float64) * dt_fs / 1000.0
    return float(1000.0 * np.polyfit(time_ps, energy_ev, 1)[0] / n_atoms)


def energy_distribution_distance_ev(
    reference_energy_ev: np.ndarray, candidate_energy_ev: np.ndarray
) -> float:
    return float(wasserstein_distance(reference_energy_ev, candidate_energy_ev))
