import numpy as np

SAMPLE_RATE = 16000


def apply_silence_mask(
    waveform: np.ndarray,
    duration_s: float,
    position: float,
    sample_rate: int = SAMPLE_RATE,
    valid_length: int | None = None,
) -> np.ndarray:
    """ Return a copy of waveform with a duration_s long window zeroed out.
    position in [0, 1] locates the start of the masked window as a fraction of
    the slide region.  """
    if not 0.0 <= position <= 1.0:
        raise ValueError(f"position must be in [0, 1], got {position}")

    n_samples = waveform.shape[-1]
    region_length = n_samples if valid_length is None else min(valid_length, n_samples)

    masked = waveform.copy()
    if duration_s <= 0:
        return masked

    mask_samples = int(round(duration_s * sample_rate))
    if mask_samples > region_length:
        raise ValueError(
            f"duration_s={duration_s} ({mask_samples} samples) exceeds the "
            f"slide region ({region_length / sample_rate:.2f}s)"
        )

    max_start = region_length - mask_samples
    start = int(round(position * max_start))
    masked[..., start : start + mask_samples] = 0.0
    return masked


def sample_duration_position(
    rng: np.random.Generator,
    min_duration_s: float = 0.0,
    max_duration_s: float = 2.0,
) -> tuple[float, float]:
    """ Draw a random (duration, position) pair for one masked sample. """
    duration = rng.uniform(min_duration_s, max_duration_s)
    position = rng.uniform(0.0, 1.0)
    return duration, position
