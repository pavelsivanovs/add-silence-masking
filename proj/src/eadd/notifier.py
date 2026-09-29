import logging

import requests
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, timeout: int = 10):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout

    def send(self, text: str):
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
        }

        try:
            response = requests.post(url, data=payload, timeout=self.timeout)
            response.raise_for_status()
        except Exception:
            logger.exception("Failed to send Telegram notification")


class TelegramCallback(TrainerCallback):
    def __init__(self, notifier, run_name: str = "training"):
        self.notifier = notifier
        self.run_name = run_name

    def on_train_begin(self, args, state, control, **kwargs):
        self.notifier.send(
            f"{self.run_name} started 🚀\n"
            f"Output dir: `{args.output_dir}`\n"
            f"Epochs: {args.num_train_epochs}\n"
            f"Train batch size/device: {args.per_device_train_batch_size}"
        )

    # def on_log(self, args, state, control, logs=None, **kwargs):
    #     if not logs:
    #         return
    #
    #     parts = [
    #         f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
    #         for k, v in logs.items()
    #     ]
    #     message = f"{self.run_name} log\nstep={state.global_step}\n" + "\n".join(parts)
    #     self.notifier.send(message)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return

        parts = [
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ]
        self.notifier.send(
            f"{self.run_name} eval 🧪\nstep={state.global_step}\n" + "\n".join(parts)
        )

    def on_train_end(self, args, state, control, **kwargs):
        self.notifier.send(
            f"{self.run_name} finished ✅\n"
            f"global_step={state.global_step}\n"
            f"best_metric={state.best_metric}\n"
        )
