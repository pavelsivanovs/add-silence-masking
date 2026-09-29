from dataclasses import dataclass

import torch
from transformers import Wav2Vec2Processor


@dataclass
class DataCollator:
    processor: Wav2Vec2Processor
    padding: bool | str = True
    max_length: int | None = None
    max_length_labels: int | None = None
    pad_to_multiple_of: int | None = None
    pad_to_multiple_of_labels: int | None = None

    def __call__(
        self, features: list[dict[str, (list[int] | torch.Tensor)]]
    ) -> dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [feature["labels"] for feature in features]
        d_type = torch.long if isinstance(label_features[0], int) else torch.float
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(label_features, dtype=d_type)
        return batch
