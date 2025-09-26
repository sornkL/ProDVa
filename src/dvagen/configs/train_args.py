from dataclasses import dataclass
from enum import Enum

from transformers import TrainingArguments


class FinetuningType(Enum):
    FULL = "full"
    LORA = "lora"
    FREEZE = "freeze"


@dataclass
class LoraArguments:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


@dataclass
class TrainArguments(TrainingArguments):
    finetuning_type: str = FinetuningType.FULL
    lora: LoraArguments | None = None
    freeze_text_encoder: bool = False
    freeze_language_model: bool = False
