from setuptools import find_packages, setup

setup(
    name="hypergraph_clustering",
    version="0.1.0",
    description="Модуль для агломеративной кластеризации гиперграфов",
    author="nosignalx2k",
    packages=find_packages(),
    install_requires=["numpy", "scikit-learn", "networkx", "matplotlib"],
)
