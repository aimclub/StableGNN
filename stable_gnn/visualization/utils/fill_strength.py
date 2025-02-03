def fill_strength(custom_scale: float, default_value: float):
    if custom_scale is None:
        return default_value
    return custom_scale * default_value
