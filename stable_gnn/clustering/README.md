# Hypergraph Clustering

## Описание
Библиотека предоставляет инструменты для агломеративной кластеризации гиперграфов. Она включает методы преобразования гиперграфов в матрицы инцидентности и смежности, автоматический выбор количества кластеров, а также метрики оценки качества кластеризации.

Кластеризация гиперграфов полезна в задачах анализа данных, таких как:
- Социальные сети
- Транспортные системы
- Биологические сети
- Анализ текстов и документов

Библиотека использует возможности `scikit-learn` и `scipy` для вычислений.

---

## Установка
1. Установите зависимости и проект:
   ```bash
   pip install -e .
   ```

---

## Использование

### Пример 1: Кластеризация гиперграфа с заданным количеством кластеров
```python
from hypergraph_clustering.utils.graph_conversion import hypergraph_to_incidence_matrix, incidence_to_adjacency
from hypergraph_clustering.clustering.agglomerative import AgglomerativeHypergraphClustering

# Пример гиперграфа
hyperedges = [[0, 1, 2], [1, 2, 3], [3, 4]]

# Преобразуем гиперграф в матрицы
incidence_matrix = hypergraph_to_incidence_matrix(hyperedges)
adjacency_matrix = incidence_to_adjacency(incidence_matrix)

# Кластеризация
clustering = AgglomerativeHypergraphClustering(n_clusters=2)
labels = clustering.fit(adjacency_matrix)

print("Кластеры:", labels)
```

### Пример 2: Автоматический выбор количества кластеров
```python
from hypergraph_clustering.clustering.auto_clustering import AutoClusterHypergraphClustering

clustering = AutoClusterHypergraphClustering(linkage="average", max_clusters=5, scoring="silhouette")
labels = clustering.fit(adjacency_matrix)

print("Кластеры:", labels)
print("Лучшее количество кластеров:", clustering.best_n_clusters)
print("Оценка:", clustering.best_score)
```

---

## Теоретическая справка

Пример гиперграфа:
- Узлы: {0, 1, 2, 3, 4}
- Гиперрёбра: {{0, 1, 2}, {1, 2, 3}, {3, 4}}

#### Преобразования гиперграфа
Для применения алгоритмов кластеризации гиперграфы преобразуются:
1. **Матрица инцидентности**: показывает связь между узлами и гиперрёбрами.
2. **Матрица смежности**: вычисляется на основе матрицы инцидентности, показывает связи между узлами.

---

### Алгоритмы кластеризации

#### Агломеративная кластеризация
Метод «снизу вверх», который начинает с представления каждого узла как отдельного кластера и объединяет их итеративно на основе заданного метода связи (`ward`, `complete`, `average`, `single`).

- **Ward**: минимизирует увеличение внутрикластерной дисперсии.
- **Complete**: учитывает максимальное расстояние между узлами.
- **Average**: основывается на среднем расстоянии между узлами.
- **Single**: минимизирует минимальное расстояние между узлами.

#### Автоматический выбор количества кластеров
Для автоматического выбора количества кластеров используется одна из метрик:
- **Silhouette**: оценивает, насколько хорошо данные сгруппированы.
- **Calinski-Harabasz**: измеряет плотность и разделение кластеров.
- **Davies-Bouldin**: оценивает близость между кластерами (меньше лучше).

---

## Структура проекта

```
hypergraph_clustering/
├── clustering/
│   ├── agglomerative.py          # Класс для агломеративной кластеризации
│   ├── auto_clustering.py        # Класс для автоматического выбора количества кластеров
│   └── __init__.py
├── metrics/
│   ├── evaluation.py             # Метрики для оценки кластеризации
│   └── __init__.py
├── tests/
│   ├── test_agglomerative.py     # Тесты для агломеративной кластеризации
│   ├── test_auto_clustering.py   # Тесты для автоматического выбора кластеров
│   ├── test_graph_conversion.py  # Тесты для преобразования гиперграфов
│   └── __init__.py
├── utils/
│   ├── graph_conversion.py       # Утилиты для работы с гиперграфами
│   ├── examples.py               # Примеры гиперграфов
│   └── __init__.py
├── README.md                     # Документация
├── setup.py                      # Файл установки
├── requirements.txt              # Зависимости
└── data/                         # Примеры гиперграфов в формате JSON
    ├── social_network.json
    ├── transport_network.json
    └── biological_network.json
```

---

## Тестирование проекта
Для проверки работоспособности проекта доступны тесты. Запустите их с помощью:
```bash
pytest hypergraph_clustering/tests/
```

---

## Источники
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Silhouette Coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering))
- [Agglomerative Clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering)

---