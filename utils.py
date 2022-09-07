import yaml

from typing import List, Dict
from argparse import ArgumentParser


def serialize_config(config: Dict) -> List[str]:
    """"""

    def parse_value(value):
        if isinstance(value, int) or isinstance(value, float) or \
           isinstance(value, str) or isinstance(value, bool):
            return str(value)
        elif isinstance(value, List):
            return [str(val) for val in value]
        else:
            raise ValueError(f"Invalid value in config file: {value}")

    # Get an empty list for serialized config:
    serialized_config = []

    for key, value in config.items():
        # Append key:
        serialized_config.append("--" + key)

        # Append value:
        if isinstance(value, bool):
            continue
        elif isinstance(value, Dict):
            serialized_config.pop()
            for k, v in value.items():
                serialized_config.append("--" + k)
                if isinstance(v, bool): continue
                serialized_config.append(parse_value(v))
        else:
            serialized_config.append(parse_value(value))

    return serialized_config


def get_distillation_args(yaml_path):
    parser = ArgumentParser()

    # Wandb
    parser.add_argument('--project', type=str,
                        help='wandb project name')
    parser.add_argument('--exp', type=str,
                        help='wandb experiement name')

    # Data configs
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--max_length', type=int, default=128,
                        help='input sequence maximum length')
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of workers for loading data")

    # Distillation configs
    parser.add_argument('--adaptors', default=[], type=list)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--temperature", default=4, type=float)

    # Optimizer configs
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)

    config = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    args = parser.parse_args(serialize_config(config))

    return args


def get_finetune_args(yaml_path):
    parser = ArgumentParser()

    # Wandb
    parser.add_argument('--project', type=str,
                        help='wandb project name')
    parser.add_argument('--exp', type=str,
                        help='wandb experiement name')

    # Dataset
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--max_length', type=int, default=128,
                        help='input sequence maximum length')
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of workers for loading data")

    # Model
    parser.add_argument("--ckpt_path", default='ckpts', type=str)

    # Training configs
    parser.add_argument("--epochs", default=5, type=int)

    # Optimizer configs
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)

    config = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    args = parser.parse_args(serialize_config(config))

    return args