from dataclasses import dataclass


@dataclass
class DataArguments:
    train_path: str
    validation_path: str
    save_train_path: str = None
    save_validation_path: str = None
    max_text_length: int = 512
    max_sequence_length: int = 512
    max_phrase_length: int = 512
    max_train_samples: int = -1
    max_eval_samples: int = -1
    overwrite_cache: bool = False
