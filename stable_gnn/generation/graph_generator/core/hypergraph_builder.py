from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx


class HypergraphBuilder:
    def __init__(self, llm_client):
        """
        Инициализация построителя гиперграфа.
        :param llm_client: Инстанс LLMClient для генерации описания гиперграфа.
        """
        self.llm_client = llm_client

    def build_hypergraph(self, text: str) -> nx.Graph:
        """
        Построение гиперграфа на основе текста.
        :param text: Входной текст для создания гиперграфа.
        :return: Объект гиперграфа (networkx.Graph).
        """
        description = self.llm_client.generate_graph_description(text)
        H = nx.Graph()  # Для гиперграфа можно использовать другие структуры, например, nx.Hypergraph

        for node, edges in description.items():
            H.add_node(node)
            for edge in edges:
                H.add_edge(node, edge)

        return H

    def build_weighted_hypergraph(self, text: str) -> nx.Graph:
        """
        Построение взвешенного гиперграфа на основе текста.
        :param text: Входной текст для создания гиперграфа.
        :return: Взвешенный гиперграф, где рёбра имеют веса.
        """
        description = self.llm_client.generate_graph_description(text)
        H = nx.Graph()

        for node, edges in description.items():
            H.add_node(node)
            for edge in edges:
                weight = self._generate_weight(node, edge)
                H.add_edge(node, edge, weight=weight)

        return H

    def _generate_weight(self, node1: str, node2: str) -> int:
        """
        Генерация веса для рёбер гиперграфа на основе узлов.
        :param node1: Первый узел.
        :param node2: Второй узел.
        :return: Взвешенное значение для рёбер.
        """
        return len(node1) + len(node2)  # Примерная логика: сумма длины строк узлов

    def visualize_hypergraph(
        self, hypergraph: nx.Graph, node_color="lightblue", edge_color="gray", node_size=500, font_size=12
    ):
        """
        Визуализация гиперграфа с настраиваемыми параметрами.
        :param hypergraph: Объект гиперграфа.
        :param node_color: Цвет узлов.
        :param edge_color: Цвет рёбер.
        :param node_size: Размер узлов.
        :param font_size: Размер шрифта для подписей.
        """
        plt.figure(figsize=(8, 6))
        nx.draw(
            hypergraph,
            with_labels=True,
            node_color=node_color,
            edge_color=edge_color,
            node_size=node_size,
            font_size=font_size,
            font_weight="bold",
            alpha=0.7,
        )
        plt.title("Гиперграф")
        plt.show()

    def save_hypergraph(self, hypergraph: nx.Graph, filename: str, format="gml"):
        """
        Сохранение гиперграфа в файл в формате GML или GraphML.
        :param hypergraph: Объект гиперграфа.
        :param filename: Имя файла для сохранения.
        :param format: Формат файла ('gml' или 'graphml').
        """
        if format == "gml":
            nx.write_gml(hypergraph, filename)
        elif format == "graphml":
            nx.write_graphml(hypergraph, filename)
        else:
            raise ValueError("Поддерживаемые форматы: 'gml', 'graphml'")

    def load_hypergraph(self, filename: str, format="gml") -> nx.Graph:
        """
        Загрузка гиперграфа из файла.
        :param filename: Имя файла для загрузки.
        :param format: Формат файла ('gml' или 'graphml').
        :return: Загруженный объект гиперграфа.
        """
        if format == "gml":
            return nx.read_gml(filename)
        elif format == "graphml":
            return nx.read_graphml(filename)
        else:
            raise ValueError("Поддерживаемые форматы: 'gml', 'graphml'")

    def describe_hypergraph(self, hypergraph: nx.Graph) -> Dict[str, List[str]]:
        """
        Описание структуры гиперграфа в виде списка гиперрёбер и связанных с ними узлов.
        :param hypergraph: Объект гиперграфа.
        :return: Структура гиперграфа, где ключи — это гиперрёбра, а значения — списки узлов.
        """
        hypergraph_description = {}

        for edge in hypergraph.edges():
            node1, node2 = edge
            if node1 not in hypergraph_description:
                hypergraph_description[node1] = []
            if node2 not in hypergraph_description[node1]:
                hypergraph_description[node1].append(node2)

            if node2 not in hypergraph_description:
                hypergraph_description[node2] = []
            if node1 not in hypergraph_description[node2]:
                hypergraph_description[node2].append(node1)

        return hypergraph_description


# Пример использования:
if __name__ == "__main__":
    from llm_client import LLMClient

    llm_client = LLMClient(model="qwen:32b")
    builder = HypergraphBuilder(llm_client)

    text = """
    Иван и Аня пошли в парк, а Петр присоединился к ним позже. В парке они встретили друзей. Все вместе они пошли в кафе.
    """

    hypergraph = builder.build_hypergraph(text)
    builder.visualize_hypergraph(hypergraph)

    builder.save_hypergraph(hypergraph, "hypergraph.gml", format="gml")
    loaded_hypergraph = builder.load_hypergraph("hypergraph.gml", format="gml")
    builder.visualize_hypergraph(loaded_hypergraph)

    weighted_hypergraph = builder.build_weighted_hypergraph(text)
    builder.visualize_hypergraph(weighted_hypergraph)

    hypergraph_description = builder.describe_hypergraph(hypergraph)
    print("Гиперграф описание:")
    print(hypergraph_description)
