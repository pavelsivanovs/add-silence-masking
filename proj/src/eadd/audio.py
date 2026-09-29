"""
Code adopted from Introduction to Speech Processing book.
https://colab.research.google.com/github/Speech-Interaction-Technology-Aalto-U/itsp/blob/main/Representations/Spectrogram_and_the_STFT.ipynb
"""

import argparse
import logging

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

logger = logging.getLogger(__name__)


TARGET_FS = 16_000


def draw_waveform(args):
    pass


def main(filename, window_length_ms, window_step_ms):
    fs, data = wavfile.read(filename)
    data = signal.resample(data, len(data) * TARGET_FS // fs)
    fs = TARGET_FS

    window_length = int(np.round(fs * window_length_ms / 1000))
    window_step = int(np.round(fs * window_step_ms / 1000))
    window_count = int(np.floor((data.shape[0] - window_length) / window_step) + 1)

    # Hann's window
    windowing_fn = (
        np.sin(
            np.pi
            * np.linspace(0.5, window_length - 0.5, num=window_length)
            / window_length
        )
        ** 2
    )

    spectrogram_matrix = np.zeros([window_length, window_count], dtype=complex)
    for window_idx in range(window_count):
        data_window = np.multiply(
            windowing_fn, data[window_idx * window_step + np.arange(window_length)]
        )
        spectrogram_matrix[:, window_idx] = np.fft.fft(data_window)

    fft_length = int((window_length + 1) / 2)

    t = np.arange(0.0, len(data), 1.0) / fs
    plt.figure(figsize=(12, 8))
    plt.subplot(311)
    plt.plot(t, data)
    plt.xlim([0, t[-1]])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Speech waveform")

    plt.subplot(312)
    plt.imshow(
        20 * np.log10(0.2 + np.abs(spectrogram_matrix[range(fft_length), :])),
        origin="lower",
        aspect="auto",
        extent=(0.0, len(data) / fs, 0.0, fs / 2000),
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.title("Speech spectrogram")

    plt.subplot(313)
    plt.imshow(
        20 * np.log10(0.2 + np.abs(spectrogram_matrix[range(fft_length // 2), :])),
        origin="lower",
        aspect="auto",
        extent=(0.0, len(data) / fs, 0.0, fs / 4000),
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.title("Speech spectrogram zoomed in to lower frequencies")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="Draw a waveform and spectrograms of an WAV file."
    )
    parser.add_argument("filename")
    parser.add_argument("--window_length_ms", type=int, default=30)
    parser.add_argument("--window_step_ms", type=int, default=5)
    args = parser.parse_args()
    main(args.filename, args.window_length_ms, args.window_step_ms)
