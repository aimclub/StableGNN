# StableGNN

<p align="center">
  <img src="https://github.com/anpolol/StableGNN/blob/main/docs/stable_gnn.png?raw=true" width="300px"> 
</p>

[![Linters](https://github.com/aimclub/StableGNN/actions/workflows/linters.yml/badge.svg)](https://github.com/aimclub/StableGNN/actions/workflows/linters.yml)
[![Tests](https://github.com/aimclub/StableGNN/actions/workflows/tests.yml/badge.svg)](https://github.com/aimclub/StableGNN/actions/workflows/tests.yml)
[![Documentation](https://github.com/aimclub/StableGNN/actions/workflows/gh_pages.yml/badge.svg)](https://aimclub.github.io/StableGNN/index.html)

StableGNN это фреймворк для автономного обучения объяснимых графовых нейронных сетей.


## Установка фреймворка
Python >= 3.9

Для начала необходимо установить [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric/) и Torch 1.1.2.

#### PyTorch 1.12

```
# CUDA 10.2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# CPU Only
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
```

Для установки PyTorch Geometric из исходных файлов для версии PyTorch 1.12.0, запустите следующие команды:

```
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html
pip install torch-geometric
```

где `${CUDA}` необходимо заменить на `cpu`, `cu102`, `cu113`, или `cu116` в зависимости от установленной версии PyTorch.

|             | `cpu` | `cu102` | `cu113` | `cu116` |
|-------------|-------|---------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      | ✅      |
| **Windows** | ✅    |         | ✅      | ✅      |
| **macOS**   | ✅    |         |         |         |


После установки Torch и Torch Geometric, склонируйте данный репозиторий и внутри него запустите команду для установки остальных библиотек:

```
pip install . 
```

## Основные элементы фреймфорка
StableGNN состоит из четырех основных частей:
* Graph: чтение входных данных и уточнение структуры графа
* ModelNodeClassification: предсказание меток вершин (задача классификации вершин) в дисассортативных графах с возможностью добавления самостоятельного обучения
* ModelGraphClassification: пердсказание меток графов (задача классификации графов) с высокой экстраполирующей способностью и с возможностью добавления самостоятельного обучения
* Explain: объяснение предсказания меток вершин

Graph состоит из следующих элементов: 
* num_nodes - число вершин в вашем графе
* y - список меток вершин, объект класса torch.Tensor; размерность (1,num_nodes)
* x - матрица аттрибутов, объект класса torch.Tensor; размерность (num_nodes,d)
* d - размерность атрибутов 
* edge_index - список рёбер, объект класса torch.Tensor; размерность (2,m), где m -- число рёбер в графе 


## Краткий обзор для новых пользователей
В первую очередь, необходимо сохранить данные в папку  
```
data_validation/dataset_name/raw
```
Папка с данными должна содержать 2 или 3 файла, если решается задача классификации вершин и N*2 файла (где N -- размер датасета), если задача классификации графов:
* **edges.txt** состоит из двух клонок, разделенных запятой; каждая строчка этого файла является парой вершин, между которыми есть ребро в графе.
* **labels.txt** колонка чисел, означающих метки вершин. Размер данной колонки равен размеру графа.
* **attrs.txt** состоит из строчек-атрибутов веришн, атрибуты отделены друг от друга запятой. Этот файл является необязательным, если входной граф не содержит атрибуты, они будут сгенерированы случайно.

Для датасета, состоящего из множества графов, требуются аналогичные файлы с постфиксом "_n.txt", где "n" -- индекс графа, кроме "labels.txt", который является одним для всего датасета.
Для уточнения структуры графа с алгоритмами уточнения, установите флаг ```adjust_flag``` на значение ```True```. Эта опция доступна только для датасетов, состоящих из одного графа (для задачи классификации вершин).

```python
from stable_gnn.graph import Graph
import torch_geometric.transforms as T

root = "../data_validation/"
name = dataset_name
adjust_flag = True 
data = Graph(name, root=root + str(dataset_name), transform=T.NormalizeFeatures(), adjust_flag=adjust_flag)[0]
```

Во фреймворке предусмотрены пайплайны тренировки моделей в модуле ```train_model_pipeline.py```. Вы можете построить собственный пайплайн наследуюясь от абстрактного класса ```TrainModel```, либо использовать готовые пайплайны для задачи классификации вершин (```TrainModelNC```) and классификации графов (```TrainModelGC```) tasks. 
```loss_name``` это название функции потерь для обучения эмбеддингов вершин без учителя, используемых в слое Geom-GCN layer, ```ssl_flag``` флаг включения функции потерь самостоятельного обучения.

```python
import torch
from stable_gnn.pipelines.train_model_pipeline import TrainModelNC, TrainModelOptunaNC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_name = 'APP'  # 'VERSE_Adj', 'LINE', 'HOPE_AA'
ssl_flag = True

optuna_training = TrainModelOptunaNC(data=data, device=device, ssl_flag=ssl_flag, loss_name=loss_name)
best_values = optuna_training.run(number_of_trials=100)
model_training = TrainModelNC(data=data, device=device, ssl_flag=ssl_flag, loss_name=loss_name)
_, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma = model_training.run(best_values)
```

Аналогичный код для задачи классификации графов за исключением нескольких параметров: ```extrapolation_flag``` флаг включения компонента экстраполяции.

```python
import torch
from stable_gnn.pipelines.train_model_pipeline import TrainModelGC, TrainModelOptunaGC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ssl_flag = True
extrapolate_flag = True

optuna_training = TrainModelOptunaGC(data=data, device=device, ssl_flag=ssl_flag, extrapolate_flag=extrapolate_flag)
best_values = optuna_training.run(number_of_trials=100)
model_training = TrainModelGC(data=data, device=device, ssl_flag=ssl_flag, extrapolate_flag=extrapolate_flag)
_, train_acc_mi, train_acc_ma, test_acc_mi, test_acc_ma = model_training.run(best_values)
```

Построение объяснений доступно только для задачи классификации вершин. После загрузки датасета с помощью класса ``Graph``, атрибуты и матрица смежности сохраняется в файлы расширения ```.npy``` и на данном этапе их необходимо загрузить.   

```python
import os
import numpy as np
from torch_geometric.utils import to_dense_adj

from stable_gnn.explain import Explain

features = np.load(root + name + "/X.npy")
if os.path.exists(root + name + "/A.npy"): 
    adj_matrix = np.load(root + name + "/A.npy")
else:
    adj_matrix = torch.squeeze(to_dense_adj(data.edge_index.cpu())).numpy()

explainer = Explain(model=model_training, adj_matrix=adj_matrix, features=features)

pgm_explanation = explainer.structure_learning(34)
assert len(pgm_explanation.nodes) >= 2
assert len(pgm_explanation.edges) >= 1
print("explanations is", pgm_explanation.nodes, pgm_explanation.edges)
```

## Обзор Архитектуры 
StableGNN это фреймворк для улучшения стабильности к шумным данным и увеличения точности на данных их разных распределений для Графовых Нейронных Сетей. Он состоит из четырех частей:
 * graph - загрузка данных и уточнение структуры
 * model_nc - модель предсказания меток вершин в графе, основанный на свертке GeomGCN, с возможностью включения функции потерь самостоятельного обучения
 * model_gc - модель предсказания меток графов с возможностью включения функции потерь самостоятельного обучения и компонента экстраполяции
 * explain - построение объяснений в виде байесовской сети  


<p align="center">
  <img src="https://github.com/anpolol/StableGNN/blob/main/docs/arch_ru.PNG?raw=true" width="800px"> 
</p>


## Сотрудничество
Чтобы внести вклад в библиотеку, необходимо следовать текущему [соглашению о коде и документации](wiki/Development.md).
Проект запускает линтеры и тесты для каждого pull request, чтобы установить линтеры и тестовые пакеты локально, запустите:

```
pip install -r requirements-dev.txt
```
Для избежания ненужных коммитов, исправляйте ошибки линтеров и тестов после запуска каждого линтера:
- `pflake8 .`
- `black .`
- `isort .`
- `mypy StableGNN`
- `pytest tests`

## Контакты
- [Связаться с командой разработчиков](mailto:egorshikov@itmo.ru)
- Natural System Simulation Team <https://itmo-nss-team.github.io/>

## Поддержка
Исследование выполнено при поддержке [Исследовательского центра сильного искусственного интеллекта в промышленности](https://sai.itmo.ru/) [Университета ИТМО](https://itmo.ru/) (Санкт-Петербург, Россия)

<p align="center">
  <img src="https://github.com/anpolol/StableGNN/blob/main/docs/AIM-logo.svg?raw=true" width="300px"> 
</p>

## Цитирование
Если используете библиотеку в ваших работах, пожалуйста, процитируйте [статью](http://www.mlgworkshop.org/2022/papers/MLG22_paper_5068.pdf) (и другие соответствующие статьи используемых методов):

```
@inproceedings{mlg2022_5068,
title={Attributed Labeled BTER-Based Generative Model for Benchmarking of Graph Neural Networks},
author={Polina Andreeva, Egor Shikov and Claudie Bocheninа},
booktitle={Proceedings of the 17th International Workshop on Mining and Learning with Graphs (MLG)},
year={2022}
}
```
