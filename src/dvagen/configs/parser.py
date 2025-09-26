import json
from dataclasses import dataclass
from typing import Any

import simple_parsing

from ..utils import logging
from .data_args import DataArguments
from .eval_args import EvalArguments
from .infer_args import InferArguments
from .model_args import DVAModelArguments
from .train_args import TrainArguments


logger = logging.get_logger(__name__)


class Args(simple_parsing.Serializable):
    # Type hints for the arguments.
    CLS = tuple[Any, ...]

    @classmethod
    def parse(cls, args: dict[str, Any] | None = None):
        assert args is None
        return simple_parsing.parse(
            cls,
            conflict_resolution=simple_parsing.ConflictResolution.NONE,  # do not allow duplicate args
            argument_generation_mode=simple_parsing.ArgumentGenerationMode.FLAT,  # default
            add_config_path_arg=True,  # allow `--config_path`
        )

    def to_json(self):
        default = lambda o: repr(o)  # default to repr if object is not serializable
        return json.dumps(self.to_dict(), indent=2, default=default)


@dataclass
class TrainArgs(Args):
    model: DVAModelArguments
    data: DataArguments
    train: TrainArguments


@dataclass
class InferArgs(Args):
    model: DVAModelArguments
    infer: InferArguments


@dataclass
class EvalArgs(Args):
    model: DVAModelArguments
    infer: InferArguments
    eval: EvalArguments


def get_train_args():
    args = TrainArgs.parse()
    logging.set_global_logger()
    logger.info_rank0(args.to_json())
    return args


def get_infer_args():
    args = InferArgs.parse()
    logging.set_global_logger()
    logger.info_rank0(args.to_json())
    return args


def get_eval_args():
    args = EvalArgs.parse()
    logging.set_global_logger()
    logger.info_rank0(args.to_json())
    return args
