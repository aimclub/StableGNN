# BellmanFordLayerModified

`BellmanFordLayer` - это PyTorch-слой, предоставляющий результаты алгоритма Беллмана-Форда для анализа графовых данных. Он возвращает матрицу расстояний и матрицу предшественников, которые могут быть использованы для поиска кратчайших путей в графе. Также этот слой определяет наличие отрицательных циклов в графе.

`BellmanFordLayerModified` - это PyTorch слой, реализующий модифицированный алгоритм Беллмана-Форда для анализа свойств графов и извлечения признаков из графовой структуры. Этот слой может использоваться в задачах графового машинного обучения, таких как предсказание путей и анализ графовых структур.

## Использование

```python
import torch
from layers.bellman_ford_modified import BellmanFordLayerModified

# Инициализация слоя с указанием количества узлов и числа признаков
num_nodes = 4
num_features = 5
bellman_ford_layer = BellmanFordLayerModified(num_nodes, num_features)

# Определение матрицы смежности графа и начального узла
adj_matrix = torch.tensor([[0, 2, float('inf'), 1],
                          [float('inf'), 0, -1, float('inf')],
                          [float('inf'), float('inf'), 0, -2],
                          [float('inf'), float('inf'), float('inf'), 0]])
source_node = 0

# Вычисление признаков графа, диаметра и эксцентриситета
node_features, diameter, eccentricity = bellman_ford_layer(adj_matrix, source_node)

print("Node Features:")
print(node_features)
print("Graph Diameter:", diameter)
print("Graph Eccentricity:", eccentricity)
```

## Параметры слоя

- `num_nodes`: Количество узлов в графе.
- `num_features`: Количество признаков, извлекаемых из графа.
- `edge_weights`: Веса ребер между узлами (обучаемые параметры).
- `node_embedding`: Вложение узлов для извлечения признаков.

## Применение:

- BellmanFordLayer полезен, когда вам нужны результаты алгоритма Беллмана-Форда для выполнения других операций или анализа графа.
- BellmanFordLayerModified полезен, когда вас помимо путей интересуют дополнительные характеристики графа, такие как диаметр и эксцентриситет.
