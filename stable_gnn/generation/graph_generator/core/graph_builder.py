import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List

class GraphBuilder:
    def __init__(self, llm_client):
        """
        Инициализация построителя графа.
        :param llm_client: Инстанс LLMClient для генерации описания графа.
        """
        self.llm_client = llm_client

    def build_graph(self, text: str) -> nx.Graph:
        """
        Построение графа на основе текста.
        :param text: Входной текст для создания графа.
        :return: Объект графа (networkx.Graph).
        """
        description = self.llm_client.generate_graph_description(text)
        G = nx.Graph()

        for node, edges in description.items():
            G.add_node(node)
            for edge in edges:
                G.add_edge(node, edge)
        
        return G

    def build_hypergraph(self, text: str) -> Dict[str, List[str]]:
        """
        Построение гиперграфа на основе текста.
        :param text: Входной текст для создания гиперграфа.
        :return: Структура гиперграфа, где ключи — это гиперрёбра, а значения — списки узлов.
        """
        description = self.llm_client.generate_hypergraph_description(text)
        hypergraph = {}

        for hyperedge, nodes in description.items():
            hypergraph[hyperedge] = nodes

        return hypergraph

    def visualize_graph(self, graph: nx.Graph, node_color='lightblue', edge_color='gray', node_size=500, font_size=12):
        """
        Визуализация графа с настраиваемыми параметрами.
        :param graph: Объект графа.
        :param node_color: Цвет узлов.
        :param edge_color: Цвет рёбер.
        :param node_size: Размер узлов.
        :param font_size: Размер шрифта для подписей.
        """
        plt.figure(figsize=(8, 6))
        nx.draw(graph, with_labels=True, node_color=node_color, edge_color=edge_color, 
                node_size=node_size, font_size=font_size, font_weight='bold', alpha=0.7)
        plt.title("Граф")
        plt.show()

    def save_graph(self, graph: nx.Graph, filename: str, format='gml'):
        """
        Сохранение графа в файл в формате GML или GraphML.
        :param graph: Объект графа.
        :param filename: Имя файла для сохранения.
        :param format: Формат файла ('gml' или 'graphml').
        """
        if format == 'gml':
            nx.write_gml(graph, filename)
        elif format == 'graphml':
            nx.write_graphml(graph, filename)
        else:
            raise ValueError("Поддерживаемые форматы: 'gml', 'graphml'")

    def load_graph(self, filename: str, format='gml') -> nx.Graph:
        """
        Загрузка графа из файла.
        :param filename: Имя файла для загрузки.
        :param format: Формат файла ('gml' или 'graphml').
        :return: Загруженный объект графа.
        """
        if format == 'gml':
            return nx.read_gml(filename)
        elif format == 'graphml':
            return nx.read_graphml(filename)
        else:
            raise ValueError("Поддерживаемые форматы: 'gml', 'graphml'")

# Пример использования:
if __name__ == "__main__":
    from llm_client import LLMClient  # Замените на путь до вашего модуля

    # Создание клиента для взаимодействия с ollama
    llm_client = LLMClient(model="qwen:32b")
    
    # Создание объекта GraphBuilder
    builder = GraphBuilder(llm_client)

    # Текст для построения графа
    text = """
    Иван и Аня пошли в парк, а Петр присоединился к ним позже. В парке они встретили друзей. Все вместе они пошли в кафе.
    """
    
    # Строим граф
    graph = builder.build_graph(text)
    
    # Визуализируем граф
    builder.visualize_graph(graph)
    
    # Сохранение графа в формат GML
    builder.save_graph(graph, 'graph.gml', format='gml')
    
    # Загрузка графа из файла
    loaded_graph = builder.load_graph('graph.gml', format='gml')

    # Визуализируем загруженный граф
    builder.visualize_graph(loaded_graph)