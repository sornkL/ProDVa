import copy
import os
import subprocess
import sys
from functools import partial

import dvagen
from dvagen.configs import get_eval_args, get_infer_args, get_train_args
from dvagen.infer.chat import chat
from dvagen.infer.eval import evaluate
from dvagen.train.train import train


USAGE = (
    f"DVAGen (version {dvagen.__version__})"
    + "\n"
    + "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   dvagen chat: Chat with the DVAModel through CLI or WebUI.        |\n"
    + "|   dvagen eval: Evaluate the DVAModel.                              |\n"
    + "|   dvagen train: Train the DVAModel.                                |\n"
    + "-" * 70
)


def parse_args(argv=None) -> dict[str, str | bool]:
    if argv is None:
        argv = sys.argv[1:]
    args = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                # --key=value
                k, v = key.split("=", 1)
                args[k] = v
            else:
                # --key value
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    args[key] = argv[i + 1]
                    i += 1
                else:
                    # --flag
                    args[key] = True
        i += 1
    return args


def main():
    COMMAND_MAP = {
        "chat": lambda: chat(get_infer_args()),
        "eval": lambda: evaluate(get_eval_args()),
        "train": lambda: train(get_train_args()),
        "help": partial(print, USAGE),
    }
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"

    if command in COMMAND_MAP:
        if command == "eval":
            results = COMMAND_MAP[command]()
            print("Evaluation Results: ")
            print(results)
        elif command == "train":
            env = copy.deepcopy(os.environ)
            args = parse_args(sys.argv)
            num_gpus = args.pop("num_gpus", 1)
            num_nodes = args.pop("num_nodes", 1)
            hostfile = args.pop("hostfile", r"")
            master_addr = args.pop("master_addr", "localhost")
            master_port = args.pop("master_port", 9901)
            train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launch_train.py")
            train_args = " ".join(f"--{k}={v}" for k, v in args.items() if v is not True)
            cmd = [
                "deepspeed",
                f"--num_gpus={num_gpus}",
                f"--num_nodes={num_nodes}",
                f"--hostfile={hostfile}",
                f"--master_addr={master_addr}",
                f"--master_port={master_port}",
                train_script,
                f"{train_args}",
            ]
            process = subprocess.run(
                cmd,
                env=env,
                check=True,
            )
            sys.exit(process.returncode)
        else:
            COMMAND_MAP[command]()
    else:
        print(f"Unknown command: {command}.\n{USAGE}")
