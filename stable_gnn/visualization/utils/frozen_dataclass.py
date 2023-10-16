from dataclasses import dataclass


def frozen_dataclass(cls):
    return dataclass(frozen=True)(cls)


def reference(cls):
    return frozen_dataclass(cls)
