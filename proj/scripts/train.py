import argparse
import logging

import numpy as np
from dotenv import load_dotenv
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    EarlyStoppingCallback,
    EvalPrediction,
    TrainingArguments,
    Wav2Vec2Config,
    Wav2Vec2Processor,
)

from eadd.collator import DataCollator
from eadd.data import get_data
from eadd.model import Wav2Vec2ForDeepfakeDetection
from eadd.notifier import TelegramCallback, TelegramNotifier
from eadd.trainer import CTCTrainer
from eadd.utils import load_config

logger = logging.getLogger(__name__)

MODEL_NAME_OR_PATH = "facebook/wav2vec2-base"

INPUT_COLUMN = "path"
OUTPUT_COLUMN = "label"


def compute_metrics(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    preds = np.argmax(logits, axis=1).squeeze()

    logits = np.asarray(logits)
    labels = np.asarray(labels).squeeze()

    # Probability of the positive class: bonafide = 1
    probs = softmax(logits, axis=1)[:, 1]

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, zero_division=0)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(np.asarray(precision).item()),
        "recall": float(np.asarray(recall).item()),
        "f1": float(np.asarray(f1).item()),
    }

    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))

    return metrics


def compute_metrics_v2(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    # Expected logits: [N, 2]
    # Make labels a flat integer vector: [N]
    labels = labels.reshape(-1).astype(np.int64)

    preds = np.argmax(logits, axis=-1).reshape(-1).astype(np.int64)
    probs = softmax(logits, axis=-1)[:, 1]

    accuracy = accuracy_score(labels, preds)

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))

    return metrics


def main(config, test: bool):
    logger.info(f"Loading model {MODEL_NAME_OR_PATH}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME_OR_PATH)
    # target_sampling_rate = processor.feature_extractor.sampling_rate
    # logger.info(f"The target sampling rate: {target_sampling_rate}. Should be 16000")

    label2id = dict(config.data.label2id)
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(label2id)

    ds = get_data(config, split="train", processor=processor, test=test)
    ds = ds.train_test_split(
        test_size=0.1,
        shuffle=True,
        stratify_by_column=config.data.train.label_col,
        seed=config.seed,
    )
    train_ds = ds["train"]
    val_ds = ds["test"]

    model_config = Wav2Vec2Config.from_pretrained(
        MODEL_NAME_OR_PATH,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        final_dropout=0.2,
        # finetuning_task="wav2vec2_clf",
    )
    setattr(model_config, "pooling_mode", config.model.pooling_mode)

    model = Wav2Vec2ForDeepfakeDetection.from_pretrained(MODEL_NAME_OR_PATH, config=model_config)

    if config.training.freeze_feature_encoder:
        logger.info("Freezing feature encoder")
        model.freeze_feature_encoder()
    if config.training.freeze_w2v2:
        logger.info("Freezing W2V2")
        model.freeze_wav2vec2()

    data_collator = DataCollator(processor=processor, padding=True)

    notifier = TelegramNotifier(config.telegram.bot_token, config.telegram.chat_id)
    telegram_callback = TelegramCallback(notifier, run_name=config.experiment)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=1,
        save_steps=0.2,
        eval_steps=0.2,
        logging_steps=0.05,
        learning_rate=1e-5,
        seed=config.seed,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics_v2,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        callbacks=[
            EarlyStoppingCallback(early_stopping_threshold=0.01),
            telegram_callback,
        ],
    )

    try:
        trainer.train()
    except Exception as e:
        notifier.send(f"Training failed\n{type(e).__name__}: {e}")
        raise e


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(prog="Train an ADD model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--test", action="store_true")

    load_dotenv()
    args = parser.parse_args()
    config = load_config(args.config)
    main(config, args.test)
