import argparse
import logging

import torch
from transformers.models.wav2vec2 import Wav2Vec2Config, Wav2Vec2Processor

from src.eadd.data import speech_file_to_array
from src.eadd.model import Wav2Vec2ForDeepfakeDetection
from src.eadd.trainer import get_device

logger = logging.getLogger(__name__)

BASE_MODEL = "facebook/wav2vec2-base"


def main(model_path, file_path):
    device = get_device()
    config = Wav2Vec2Config.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model = Wav2Vec2ForDeepfakeDetection.from_pretrained(model_path).to(device)

    speech = speech_file_to_array(file_path)

    features = processor(
        speech,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    input_values = features.input_values.to(device)
    logger.debug(f"input_values: {input_values}")
    with torch.no_grad():
        logits = model(input_values).logits
    scores = torch.sigmoid(logits).detach().cpu().numpy()
    real_score = scores[0][config.label2id["real"]]
    fake_score = scores[0][config.label2id["generated"]]
    logger.info(f"real_score: {real_score}")
    logger.info(f"fake_score: {fake_score}")


if __name__ == "__main__":
    logging.basicConfig(
        # level=logging.DEBUG,
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    )

    parser = argparse.ArgumentParser("Analyze single WAV file.")
    parser.add_argument("model_path")
    parser.add_argument("file_path")
    args = parser.parse_args()

    main(args.model_path, args.file_path)
