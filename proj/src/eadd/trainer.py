from typing import Any, Dict, Union

import torch
from torch import nn
from transformers import Trainer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


device = get_device()


class CTCTrainer(Trainer):
    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch,
    ) -> torch.Tensor:
        model.to(device)
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        return loss.detach()
