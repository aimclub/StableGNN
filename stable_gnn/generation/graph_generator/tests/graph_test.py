import pytest
from unittest.mock import MagicMock
import networkx as nx
from stable_gnn.generation.graph_generator.core.graph_builder import GraphBuilder

@pytest.fixture
def llm_client_mock():
    """Фикстура для мокированного клиента LLM."""
    llm_client = MagicMock()
    llm_client.generate_graph_description.return_value = {
        "Иван": ["Аня", "Петр"],
        "Аня": ["Иван", "Петр"],
        "Петр": ["Иван", "Аня"]
    }
    llm_client.generate_hypergraph_description.return_value = {
        "hyperedge_1": ["Иван", "Аня", "Петр"]
    }
    return llm_client

@pytest.fixture
def graph_builder(llm_client_mock):
    """Фикстура для инициализации GraphBuilder."""
    return GraphBuilder(llm_client=llm_client_mock)

def test_build_graph(graph_builder, llm_client_mock):
    """Тестирование метода build_graph."""
    text = "Иван и Аня пошли в парк, а Петр присоединился к ним позже."
    graph = graph_builder.build_graph(text)
    
    # Проверка, что граф был построен правильно
    assert len(graph.nodes) == 3  # Иван, Аня, Петр
    assert len(graph.edges) == 3   # Иван-Aня, Иван-Петр, Аня-Петр
    
    # Проверка, что метод LLMClient был вызван
    llm_client_mock.generate_graph_description.assert_called_once_with(text)

def test_build_hypergraph(graph_builder, llm_client_mock):
    """Тестирование метода build_hypergraph."""
    text = "Иван, Аня и Петр пошли в парк, а затем все пошли в кафе."
    hypergraph = graph_builder.build_hypergraph(text)
    
    # Проверка, что гиперграф построен правильно
    assert len(hypergraph) == 1  # 1 гиперрёбро
    assert hypergraph["hyperedge_1"] == ["Иван", "Аня", "Петр"]
    
    # Проверка, что метод LLMClient был вызван
    llm_client_mock.generate_hypergraph_description.assert_called_once_with(text)

def test_visualize_graph(graph_builder):
    """Тестирование метода visualize_graph (непосредственное тестирование визуализации сложно, проверим без ошибок)."""
    graph = nx.Graph()
    graph.add_nodes_from(["Иван", "Аня", "Петр"])
    graph.add_edges_from([("Иван", "Аня"), ("Иван", "Петр"), ("Аня", "Петр")])
    
    # Проверка, что метод visualize_graph выполняется без ошибок
    try:
        graph_builder.visualize_graph(graph)
    except Exception as e:
        pytest.fail(f"Ошибка при визуализации графа: {e}")

def test_save_and_load_graph(graph_builder, tmp_path):
    """Тестирование сохранения и загрузки графа."""
    graph = nx.Graph()
    graph.add_nodes_from(["Иван", "Аня", "Петр"])
    graph.add_edges_from([("Иван", "Аня"), ("Иван", "Петр"), ("Аня", "Петр")])
    
    # Сохранение графа в GML
    file_path = tmp_path / "graph.gml"
    graph_builder.save_graph(graph, file_path, format="gml")
    
    # Загрузка графа из файла
    loaded_graph = graph_builder.load_graph(file_path, format="gml")
    
    # Проверка, что граф был правильно загружен
    assert len(loaded_graph.nodes) == 3
    assert len(loaded_graph.edges) == 3

@pytest.mark.parametrize("format", ["gml", "graphml"])
def test_save_graph_invalid_format(graph_builder, tmp_path, format):
    """Тестирование сохранения графа с некорректным форматом."""
    graph = nx.Graph()
    graph.add_nodes_from(["Иван", "Аня", "Петр"])
    graph.add_edges_from([("Иван", "Аня"), ("Иван", "Петр"), ("Аня", "Петр")])
    
    with pytest.raises(ValueError):
        graph_builder.save_graph(graph, tmp_path / "graph.invalid", format="invalid_format")

