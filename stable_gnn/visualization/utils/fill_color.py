from typing import Any

from stable_gnn.visualization.exceptions.exceptions_classes import ParamsValidationException


def fill_color(custom_color: str | list, default_color: Any, length: int):
    if custom_color is None:
        return [default_color] * length

    elif isinstance(custom_color, list):
        if isinstance(custom_color[0], str) or isinstance(custom_color[0], tuple) or isinstance(custom_color[0], list):
            return custom_color
        else:
            return [custom_color] * length

    elif isinstance(custom_color, str):
        return [custom_color] * length

    else:
        raise ParamsValidationException("The specified value is not a valid type.")
