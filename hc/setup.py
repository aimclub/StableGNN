from setuptools import setup, find_packages
from typing import Dict

version: Dict[str, str] = {}
with open("requirements.txt", "r") as f:
    requirements = [x for x in f.read().splitlines() if "#" not in x]

setup(
    name="hypergraph_clustering",
    version="0.1.0",
    description="Модуль для агломеративной кластеризации гиперграфов",
    author="nosignalx2k",
    packages=find_packages(),
    install_requires=requirements,
)