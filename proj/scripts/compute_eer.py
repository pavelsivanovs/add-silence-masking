import argparse
import logging

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import classification_report, roc_curve
from transformers import Wav2Vec2Config, Wav2Vec2Processor

import eadd.data as eadd_data
from eadd.data import get_data, resolve_audio_path_relative_to_data_file
from eadd.model import Wav2Vec2ForDeepfakeDetection
from eadd.trainer import get_device
from eadd.utils import load_config

logger = logging.getLogger(__name__)

BASE_MODEL = "facebook/wav2vec2-base"


def speech_file_to_array(path: str) -> np.ndarray:
    speech, _ = sf.read(path, dtype="float32", always_2d=True)
    speech = speech.T  # (channels, samples)
    max_samples = eadd_data.AUDIO_LENGTH_LIMIT_IN_SECONDS * eadd_data.SAMPLE_RATE
    if speech.shape[1] < max_samples:
        speech = np.pad(speech, ((0, 0), (0, max_samples - speech.shape[1])))
    else:
        speech = speech[:, :max_samples]
    if speech.shape[0] == 1:
        speech = speech[0]
    return speech


eadd_data.speech_file_to_array = speech_file_to_array


def get_preprocess(dataset_root, input_column):
    def preprocess(batch):
        absolute_paths = [
            resolve_audio_path_relative_to_data_file(dataset_root, path) for path in batch[input_column]
        ]
        speech_list = [speech_file_to_array(absolute_path) for absolute_path in absolute_paths]
        batch["speech"] = speech_list
        return batch

    return preprocess


def get_predict(model, processor, device):
    def predict(batch):
        features = processor(
            batch["speech"],
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = features.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        batch["predicted"] = pred_ids
        batch["logits"] = logits.detach().cpu().numpy()
        return batch

    return predict


def compute_eer(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx]
    return eer, eer_threshold


def main(config, limit, output_path):
    logger.info(f"Initializing EER computation of model {config.output_dir} with data from {config.data.test.data_root}")

    device = get_device()
    checkpoint_path = config.output_dir + "/" + config.best_checkpoint
    model_config = Wav2Vec2Config.from_pretrained(checkpoint_path)
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model = Wav2Vec2ForDeepfakeDetection.from_pretrained(checkpoint_path).to(device)
    model.eval()

    test_dataset = get_data(config, split="test", processor=processor, test=False, limit=limit)
    logger.info(f"Loaded dataset of size {test_dataset.num_rows}")

    preprocess = get_preprocess(config.data.test.data_root, config.data.test.pathfile_col)
    predict = get_predict(model, processor, device)
    logger.info("Preprocessing data.")
    test_dataset = test_dataset.map(preprocess, batched=True, batch_size=8, num_proc=8)
    results = test_dataset.map(predict, batched=True, batch_size=4)

    label_ids = list(range(model_config.num_labels))
    label_names = [str(model_config.id2label[i]) for i in label_ids]
    y_true = np.array(results[config.data.test.label_col])
    y_pred = np.array(results["predicted"])

    report = classification_report(y_true, y_pred, labels=label_ids, target_names=label_names, digits=4)
    print(report)

    logits = np.array(results["logits"]).reshape(-1, model_config.num_labels)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    bonafide_id = model_config.label2id["bonafide"]
    bonafide_prob = probs[:, bonafide_id]
    is_bonafide = (y_true == bonafide_id).astype(int)

    eer, eer_threshold = compute_eer(is_bonafide, bonafide_prob)
    print(f"EER: {eer * 100:.2f}% (threshold on bonafide probability: {eer_threshold:.4f})")

    if output_path:
        import pandas as pd

        df = pd.DataFrame(
            {
                "true_label": [model_config.id2label[i] for i in y_true],
                "predicted_label": [model_config.id2label[i] for i in y_pred],
                "bonafide_prob": bonafide_prob,
            }
        )
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Wrote per-utterance scores to {output_path}")

    return eer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(prog="Compute EER for an ADD model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate on only the first N rows of the test split.")
    parser.add_argument("--output", type=str, default=None, help="Path to write per-utterance scores (TSV).")

    args = parser.parse_args()
    config = load_config(args.config)
    main(config, args.limit, args.output)
