# Docker Devbox

Requirements:

* Docker
* Docker Compose
* Docker Nvidia Runtime

## How to run

Copy docker-compose from example:

```shell
cp docker-compose.dist.yml docker-compose.yml
```

Build an image:

```shell
docker-compose build
```

Start a container:

```shell
docker-compose up -d
```

## How to use inside a container

Login into a container:

```shell
docker-compose exec app bash
```

Use python3.9 interpreter for running all your tasks, e.g. tutorials:

```shell
# Prepare environment
cd tutorials
ln -s ../stable_gnn

# Run graph classification task
python3.9 graph_classification.py

# Run node classification task
python3.9 node_classification.py
```

## How to use from IDE like PyCharm

You need to `Add New Interpreter` via menu of PyCharn, then select `Docker Compose`