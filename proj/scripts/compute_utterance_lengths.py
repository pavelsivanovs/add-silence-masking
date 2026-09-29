import argparse
import sys
import time

import pandas as pd
import soundfile as sf

from eadd.data import AUDIO_LENGTH_LIMIT_IN_SECONDS, SAMPLE_RATE

MAX_SAMPLES = AUDIO_LENGTH_LIMIT_IN_SECONDS * SAMPLE_RATE


def compute_lengths(data_root: str, audio_file_names: list[str]) -> pd.DataFrame:
    t0 = time.time()
    lengths = {}
    for i, fname in enumerate(sorted(set(audio_file_names))):
        info = sf.info(f"{data_root}/{fname}")
        frames = min(info.frames, MAX_SAMPLES)
        lengths[fname] = frames / info.samplerate
        if i % 1000 == 0:
            print(f"{i}/{len(audio_file_names)} elapsed={time.time() - t0:.1f}s", file=sys.stderr)
    return pd.DataFrame(
        {"audio_file_name": list(lengths.keys()), "utterance_length_s": list(lengths.values())}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Compute per-utterance audio lengths.")
    parser.add_argument(
        "--data-root",
        default="data/asv2019_la/ASVspoof2019_LA_eval/flac",
        help="Directory containing the eval-split flac files.",
    )
    parser.add_argument(
        "--tsvs",
        nargs="+",
        default=[
            "artifacts/silence_experiment_default_mean.tsv",
            "artifacts/silence_experiment_default_max.tsv",
        ],
        help="Silence-experiment TSVs to pull audio_file_name from.",
    )
    parser.add_argument("--output", default="artifacts/utterance_lengths.tsv")
    args = parser.parse_args()

    audio_file_names: list[str] = []
    for tsv in args.tsvs:
        audio_file_names.extend(pd.read_csv(tsv, sep="\t")["audio_file_name"].tolist())

    out = compute_lengths(args.data_root, audio_file_names)
    out.to_csv(args.output, sep="\t", index=False)
    print(f"wrote {len(out)} rows to {args.output}", file=sys.stderr)
