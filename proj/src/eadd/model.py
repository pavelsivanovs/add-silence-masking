from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.file_utils import ModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    pooled_output: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None


class Wav2VecClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def masked_mean(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def masked_sum(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1)


def masked_max(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).bool()
    masked = hidden_states.masked_fill(~mask, float("-inf"))
    # return masked.max(dim=1).values
    return masked.max(dim=1)


class Wav2Vec2ForDeepfakeDetection(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2VecClassificationHead(config)
        self.loss = nn.CrossEntropyLoss()

        self.merge_fn_map = {
            "mean": self._merge_mean,
            "sum": self._merge_sum,
            "max": self._merge_max,
        }

        self.post_init()

    @staticmethod
    def _merge_mean(hidden_states, attention_mask=None):
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        return masked_mean(hidden_states, attention_mask)

    @staticmethod
    def _merge_sum(hidden_states, attention_mask=None):
        if attention_mask is None:
            return hidden_states.sum(dim=1)
        return masked_sum(hidden_states, attention_mask)

    @staticmethod
    def _merge_max(hidden_states, attention_mask=None):
        if attention_mask is None:
            return hidden_states.max(dim=1).values
        return masked_max(hidden_states, attention_mask)

    def freeze_feature_encoder(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_wav2vec2(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def merge(self, hidden_states, attention_mask, mode="mean"):
        """Merge hidden states into a single vector."""
        if mode not in self.merge_fn_map:
            raise KeyError(f"Unknown merge mode: {mode}")
        return self.merge_fn_map[mode](hidden_states, attention_mask)

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        # return_dict = (
        #     return_dict if return_dict is not None else self.config.use_return_dict
        # )
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.merge(last_hidden_state, attention_mask, self.pooling_mode)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            assert labels.dtype == torch.long
            loss = self.loss(logits, labels)

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            pooled_output=pooled_output,
            last_hidden_state=last_hidden_state,
        )
