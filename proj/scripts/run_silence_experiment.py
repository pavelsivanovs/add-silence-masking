import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2Config, Wav2Vec2Processor

from eadd.data import (
    AUDIO_LENGTH_LIMIT_IN_SECONDS,
    SAMPLE_RATE,
    resolve_audio_path_relative_to_data_file,
)
from eadd.masking import apply_silence_mask, sample_duration_position
from eadd.model import Wav2Vec2ForDeepfakeDetection
from eadd.trainer import get_device
from eadd.utils import load_config

logger = logging.getLogger(__name__)

BASE_MODEL = "facebook/wav2vec2-base"
MAX_SAMPLES = AUDIO_LENGTH_LIMIT_IN_SECONDS * SAMPLE_RATE


def load_and_pad(path: str) -> tuple[np.ndarray, int]:
    speech, _ = torchaudio.load(path)
    if speech.shape[0] > 1:
        speech = speech[:1]

    original_length = speech.shape[1]
    if original_length < MAX_SAMPLES:
        speech = F.pad(speech, (0, MAX_SAMPLES - original_length))
        valid_length = original_length
    else:
        speech = speech[:, :MAX_SAMPLES]
        valid_length = MAX_SAMPLES

    return speech[0].numpy(), valid_length


def run_experiment(config, limit, n_draws, max_duration_s, seed, batch_size, output_path):
    device = get_device()
    checkpoint_path = f"{config.output_dir}/{config.best_checkpoint}"
    model_config = Wav2Vec2Config.from_pretrained(checkpoint_path)
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model = Wav2Vec2ForDeepfakeDetection.from_pretrained(checkpoint_path).to(device)
    model.eval()

    test_config = config.data.test
    df = pd.read_csv(test_config.data_file, sep="\t")
    if limit is not None:
        df = df.head(limit)
    logger.info(f"Running silence experiment on {len(df)} utterances x {n_draws} draw(s) each.")

    rng = np.random.default_rng(seed if seed is not None else config.seed)
    bonafide_id = config.data.label2id["bonafide"]

    if os.path.exists(output_path):
        os.remove(output_path)
    rows_written = 0

    batch_waveforms = []
    batch_meta = []

    def flush_batch():
        nonlocal rows_written
        if not batch_waveforms:
            return
        features = processor(batch_waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = features.input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(logits.cpu().numpy(), axis=-1)
        batch_records = []
        for meta, pred, prob in zip(batch_meta, preds, probs):
            predicted_label = model_config.id2label[int(pred)]
            batch_records.append(
                {
                    **meta,
                    "predicted_label": predicted_label,
                    "correct": int(predicted_label == meta["true_label"]),
                    "prob_bonafide": float(prob[bonafide_id]),
                }
            )
        pd.DataFrame.from_records(batch_records).to_csv(
            output_path, sep="\t", index=False, mode="a", header=(rows_written == 0)
        )
        rows_written += len(batch_records)
        batch_waveforms.clear()
        batch_meta.clear()

    for row in df.itertuples():
        audio_file_name = getattr(row, test_config.pathfile_col)
        true_label = getattr(row, test_config.label_col)
        speaker_id = row.speaker_id

        path = resolve_audio_path_relative_to_data_file(test_config.data_root, audio_file_name)
        waveform, valid_length = load_and_pad(path)

        for draw_idx in range(n_draws):
            duration_s, position = sample_duration_position(rng, max_duration_s=max_duration_s)
            # Keep the mask comfortably inside the real speech region.
            duration_s = min(duration_s, max(valid_length / SAMPLE_RATE - 0.05, 0.0))

            masked = apply_silence_mask(waveform, duration_s=duration_s, position=position, valid_length=valid_length)
            batch_waveforms.append(masked)
            batch_meta.append(
                {
                    "speaker_id": speaker_id,
                    "audio_file_name": audio_file_name,
                    "true_label": true_label,
                    "duration_s": duration_s,
                    "position": position,
                    "draw_idx": draw_idx,
                }
            )
            if len(batch_waveforms) >= batch_size:
                flush_batch()

    flush_batch()

    logger.info(f"Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(prog="Run the silence-masking experiment against an ADD model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--limit", type=int, default=None, help="Number of utterances to sample from the eval split.")
    parser.add_argument("--n-draws", type=int, default=1, help="Number of (duration, position) draws per utterance.")
    parser.add_argument("--max-duration", type=float, default=2.0, help="Upper bound (s) for the duration draw.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed; defaults to config.seed.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=str, required=True, help="Output TSV path.")

    args = parser.parse_args()
    config = load_config(args.config)
    run_experiment(
        config,
        args.limit,
        args.n_draws,
        args.max_duration,
        args.seed,
        args.batch_size,
        args.output,
    )
