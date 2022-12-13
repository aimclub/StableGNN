try:
    import torch
    from torch_geometric.typing import Tensor
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "You need install torch and torch geometric first. See https://github.com/anpolol/StableGNN#installation for deatails")

from typing import Dict

from setuptools import find_packages, setup

version: Dict[str, str] = {}
with open("requirements.txt", "r") as f:
    requirements = [x for x in f.read().splitlines() if "#" not in x]

with open("stable_gnn/version.py") as f:
    exec(f.read(), version)

setup(
    name="stable_gnn",
    packages=find_packages(exclude="tests"),
    include_package_data=True,
    version=version["version"],
    description="StableGNN library",
    author="NCCR Team, ITMO Univerisy",
    install_requires=requirements,
)
