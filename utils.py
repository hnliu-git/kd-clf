import yaml

from typing import List, Dict
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification
)


def path_to_clf_model(path: str, num_classes: int):
    if 'yaml' in path:
        args = yaml.load(open(path), Loader=yaml.FullLoader)
        config = BertConfig(**args, num_labels=num_classes)
        model = BertForSequenceClassification(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_classes)

    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.config.output_values = True

    return model


def path_to_mlm_model(path: str):
    if 'yaml' in path:
        args = yaml.load(open(path), Loader=yaml.FullLoader)
        config = BertConfig(**args)
        model = BertForMaskedLM(config)
    else:
        model = BertForMaskedLM.from_pretrained(path)

    return model


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