import argparse
import logging

import numpy as np
import torch
import tqdm
from datasets import load_dataset
from transformers import Wav2Vec2Config, Wav2Vec2Processor

from eadd.data import resolve_audio_path_relative_to_data_file, speech_file_to_array
from eadd.model import Wav2Vec2ForDeepfakeDetection
from eadd.trainer import get_device

logger = logging.getLogger(__name__)

BASE_MODEL = "facebook/wav2vec2-base"


def get_preprocess_fn(dataset_path):
    def preprocess(batch):
        absolute_paths = [
            resolve_audio_path_relative_to_data_file(dataset_path, path)
            for path in batch["path"]
        ]
        speech_list = [
            speech_file_to_array(absolute_path) for absolute_path in absolute_paths
        ]
        batch["speech"] = speech_list
        return batch

    return preprocess


def get_predict_fn(model, processor, device):
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


def get_predictions(model_path, dataset_path, name, n, test, output_dir):
    ds = load_dataset("csv", data_files=dataset_path, delimiter="\t")["train"]
    if test:
        ds = ds.select(range(32))
    logger.info(f"Loaded dataset from {dataset_path}: {ds}")

    device = get_device()
    config = Wav2Vec2Config.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model = Wav2Vec2ForDeepfakeDetection.from_pretrained(model_path).to(device)
    model.eval()
    logger.info("Loaded model.")

    preprocess = get_preprocess_fn(dataset_path)
    predict = get_predict_fn(model, processor, device)
    ds = ds.map(preprocess, batched=True, batch_size=8, num_proc=8)
    logger.info("Preprocessed data.")
    for i in tqdm.tqdm(range(n), desc="Predicting"):
        predictions = ds.map(predict, batched=True, batch_size=4).to_pandas()
        # Labels are "bonafide"/"spoof" (see config.data.label2id), not "real"/"generated".
        bonafide_id = config.label2id["bonafide"]
        logits = predictions["logits"].agg(np.concat).reshape(-1, 2)
        logits = torch.tensor(logits)
        real_prob = torch.softmax(logits, dim=-1).numpy()[:, bonafide_id]
        y_true = [config.label2id[name] for name in predictions["label"]]
        predictions["true"] = y_true
        predictions["real_prob"] = real_prob

        df = predictions[
            ["name", "path", "duration", "label", "true", "predicted", "real_prob"]
        ]
        df.to_csv(f"{output_dir}/{name}.{i}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s"
    )

    parser = argparse.ArgumentParser(
        prog="Calculate EER for a Wav2Vec2ForDeepfakeDetection model given the test dataset."
    )
    parser.add_argument("model_path")
    parser.add_argument("dataset_path")
    parser.add_argument("name")
    parser.add_argument("--n", default=3, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output_dir", default="predictions")
    args = parser.parse_args()

    get_predictions(
        args.model_path,
        args.dataset_path,
        args.name,
        args.n,
        args.test,
        args.output_dir,
    )
