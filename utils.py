from typing import List, Dict


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