from dataclasses import dataclass
from enum import Enum


class EvalTaskType(Enum):
    LANGUAGE_MODELING = "LANGUAGE_MODELING"
    PROTEIN_DESIGN = "PROTEIN_DESIGN"


@dataclass
class EvalArguments:
    test_data_file: str
    batch_size: int
    task_type: EvalTaskType
    eval_seed: int | None = None
    save_results_path: str | None = None
    prefix_tokenizer_path: str | None = None
    prefix_tokens: int = 32
    mauve_model_path: str | None = None
    mauve_batch_size: int = 1
    perplexity_model_path: str | None = None
    perplexity_batch_size: int = 1
    nsl_tokenizer_path: str | None = None
