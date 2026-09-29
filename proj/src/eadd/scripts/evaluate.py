import argparse
import logging

import torch
from sklearn.metrics import classification_report
from transformers import Wav2Vec2Config, Wav2Vec2Processor

from eadd.data import get_data, resolve_audio_path_relative_to_data_file, speech_file_to_array
from eadd.model import Wav2Vec2ForDeepfakeDetection
from eadd.trainer import get_device
from eadd.utils import load_config

logger = logging.getLogger(__name__)

BASE_MODEL = "facebook/wav2vec2-base"


def get_preprocess(dataset_root, input_column):
    def preprocess(batch):
        absolute_paths = [resolve_audio_path_relative_to_data_file(dataset_root, path) for path in batch[input_column]]
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


"""
I need to load the test data, and run the model against it.
"""


def main(config, small, limit):
    logger.info(f"Initializing evaluation of model {config.output_dir} with data from {config.data.test.data_root}")

    device = get_device()
    checkpoint_path = config.output_dir + "/" + config.best_checkpoint
    model_config = Wav2Vec2Config.from_pretrained(checkpoint_path)
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model = Wav2Vec2ForDeepfakeDetection.from_pretrained(checkpoint_path).to(device)

    # test_dataset = load_dataset("csv", data_files=args.test_dataset, delimiter="\t")["train"]

    test_dataset = get_data(config, split="test", processor=processor, test=True if small else False, limit=limit)
    logger.info(f"Loaded dataset of size {test_dataset.num_rows}")

    preprocess = get_preprocess(config.data.test.data_root, config.data.test.pathfile_col)
    predict = get_predict(model, processor, device)
    logger.info("Preprocessing data.")
    test_dataset = test_dataset.map(preprocess, batched=True, batch_size=8, num_proc=8)
    results = test_dataset.map(predict, batched=True, batch_size=4)

    label_ids = list(range(model_config.num_labels))
    label_names = [str(model_config.id2label[i]) for i in label_ids]
    # get_data() already casts the label column to ClassLabel ints matching label2id.
    y_true = list(results[config.data.test.label_col])
    y_pred = results["predicted"]

    report = classification_report(y_true, y_pred, labels=label_ids, target_names=label_names, digits=4)
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(prog="Evaluate an ADD model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--small", action="store_true", help="Evaluate on the first 100 rows (quick smoke test).")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate on only the first N rows of the test split.")

    args = parser.parse_args()
    config = load_config(args.config)
    main(config, args.small, args.limit)
