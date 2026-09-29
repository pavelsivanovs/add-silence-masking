import logging
from pathlib import Path

import numpy as np
import torch.nn.functional as F
import torchaudio
from datasets import ClassLabel, load_dataset

logger = logging.getLogger(__name__)


def resolve_audio_path_relative_to_data_file(data_root: str, audio_file_path: str) -> str:
    data_root_path = Path(data_root)
    audio_path = Path(audio_file_path)
    return str(data_root_path / audio_path)


AUDIO_LENGTH_LIMIT_IN_SECONDS = 10
SAMPLE_RATE = 16000


def speech_file_to_array(path: str) -> np.ndarray:
    speech, _ = torchaudio.load(path)
    max_samples = AUDIO_LENGTH_LIMIT_IN_SECONDS * SAMPLE_RATE
    if speech.shape[1] < max_samples:
        speech = F.pad(speech, (0, max_samples - speech.shape[1]))
    else:
        speech = speech[:, :max_samples]

    if speech.shape[0] == 1:
        speech = speech[0]

    return speech.numpy()


def preprocess_dataset(
    dataset,
    input_column: str,
    output_column,
    processor,
    data_root,
):
    def preprocess_batch(samples):
        absolute_paths = [resolve_audio_path_relative_to_data_file(data_root, path) for path in samples[input_column]]
        speech_list = [speech_file_to_array(absolute_path) for absolute_path in absolute_paths]
        target_list = [label for label in samples[output_column]]
        result = processor(speech_list, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        result["labels"] = list(target_list)
        return result

    dataset = dataset.map(preprocess_batch, batch_size=64, batched=True, num_proc=8)
    return dataset


def get_data(config, split: str, processor, test, limit: int | None = None):
    split_config = config.data[split]
    # Making sure that strings will be converted to the exact same values as defined in label2id
    label2id = config.data.label2id
    labels = [k for k, _ in sorted(label2id.items(), key=lambda item: item[1])]

    if limit is not None:
        split_arg = f"train[:{limit}]"
    elif test:
        split_arg = "train[:100]"
    else:
        split_arg = "train"

    ds = load_dataset(
        path="csv",
        data_files=split_config.data_file,
        delimiter="\t",
        split=split_arg,
        num_proc=8,
    )

    def encode_label(row):
        row[split_config.label_col] = label2id[row[split_config.label_col]]
        return row

    ds = ds.map(encode_label)
    ds = ds.cast_column(split_config.label_col, ClassLabel(names=labels))

    label_feature = ds.features[split_config.label_col]
    logger.debug(f"{label_feature.names}: {label_feature.int2str(0)}")

    ds = preprocess_dataset(
        ds,
        split_config.pathfile_col,
        split_config.label_col,
        processor,
        split_config.data_root,
    )

    return ds


def old_get_data(dataset_path, input_column, output_column, target_sampling_rate, processor, seed=42):
    ds = load_dataset("csv", data_files=dataset_path, delimiter="\t", split="train")
    # this changes the value of the output_column
    # as we can see, it mapped generated to 0 and real to 1, but cannot be sure
    ds = ds.class_encode_column(output_column)
    ds = ds.train_test_split(test_size=0.1, shuffle=True, stratify_by_column=output_column, seed=seed)
    train_ds = ds["train"]
    test_ds = ds["test"]

    # label_list = train_ds.unique(output_column)
    label_list = train_ds.features[output_column].names
    num_labels = len(label_list)
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for idx, label in enumerate(label_list)}
    logger.debug(f"Label list: {label_list}")
    logger.debug(f"Number of labels: {num_labels}")
    logger.debug(f"label2id: {label2id}")
    logger.debug(f"id2label: {id2label}")

    train_ds = preprocess_dataset(
        train_ds,
        input_column,
        output_column,
        target_sampling_rate,
        processor,
        dataset_path,
    )
    test_ds = preprocess_dataset(
        test_ds,
        input_column,
        output_column,
        target_sampling_rate,
        processor,
        label2id,
        dataset_path,
    )

    return {
        "train_ds": train_ds,
        "test_ds": test_ds,
        "label_list": label_list,
        "num_labels": num_labels,
        "label2id": label2id,
        "id2label": id2label,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    # get_data(
    #     dataset_path="data/small_drssu_train.tsv",
    #     input_column="path",
    #     output_column="label",
    #     target_sampling_rate=16000,
    #     processor=None,
    # )
    # # get_data(dataset_path='data/small_asv_train.tsv', input_column='path', output_column='label', target_sampling_rate=16000, processor=None)
    # # get_data(dataset_path='data/small_mixed_train.tsv', input_column='path', output_column='label', target_sampling_rate=16000, processor=None)
