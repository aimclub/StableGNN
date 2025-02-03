from dataclasses import dataclass


def frozen_dataclass(cls):
    """Frozen dataclass decorator for dataclass based immutable enums."""
    return dataclass(frozen=True)(cls)


def reference(cls):
    """Frozen dataclass decorator for dataclass based immutable enums (renamed, verbose)."""
    return frozen_dataclass(cls)
