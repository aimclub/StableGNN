import pytest
from unittest.mock import MagicMock
import networkx as nx
from stable_gnn.generation.graph_generator.core.hypergraph_builder import HypergraphBuilder

@pytest.fixture
def llm_client_mock():
    """Фикстура для мокированного клиента LLM."""
    llm_client = MagicMock()
    llm_client.generate_graph_description.return_value = {
        "Иван": ["Аня", "Петр"],
        "Аня": ["Иван", "Петр"],
        "Петр": ["Иван", "Аня"]
    }
    return llm_client

@pytest.fixture
def hypergraph_builder(llm_client_mock):
    """Фикстура для инициализации HypergraphBuilder."""
    return HypergraphBuilder(llm_client=llm_client_mock)

def test_build_hypergraph(hypergraph_builder, llm_client_mock):
    """Тестирование метода build_hypergraph."""
    text = "Иван и Аня пошли в парк, а Петр присоединился к ним позже."
    hypergraph = hypergraph_builder.build_hypergraph(text)
    
    # Проверка, что гиперграф был построен правильно
    assert len(hypergraph.nodes) == 3  # Иван, Аня, Петр
    assert len(hypergraph.edges) == 3   # Иван-Аня, Иван-Петр, Аня-Петр
    
    # Проверка, что метод LLMClient был вызван
    llm_client_mock.generate_graph_description.assert_called_once_with(text)

def test_build_weighted_hypergraph(hypergraph_builder, llm_client_mock):
    """Тестирование метода build_weighted_hypergraph."""
    text = "Иван, Аня и Петр пошли в парк, а затем все пошли в кафе."
    weighted_hypergraph = hypergraph_builder.build_weighted_hypergraph(text)
    
    # Проверка, что гиперграф построен правильно с весами
    assert len(weighted_hypergraph.nodes) == 3  # Иван, Аня, Петр
    assert len(weighted_hypergraph.edges) == 3   # Иван-Аня, Иван-Петр, Аня-Петр
    
    # Проверка, что рёбра имеют вес
    for u, v, data in weighted_hypergraph.edges(data=True):
        assert "weight" in data  # Проверка наличия веса у рёбер
    
    llm_client_mock.generate_graph_description.assert_called_once_with(text)

def test_generate_weight(hypergraph_builder):
    """Тестирование метода _generate_weight."""
    weight = hypergraph_builder._generate_weight("Иван", "Аня")
    
    # Проверка правильности веса (сумма длины строк)
    assert weight == len("Иван") + len("Аня")  # 4 + 3 = 7

def test_visualize_hypergraph(hypergraph_builder):
    """Тестирование метода visualize_hypergraph (непосредственное тестирование визуализации сложно, проверим без ошибок)."""
    hypergraph = nx.Graph()
    hypergraph.add_nodes_from(["Иван", "Аня", "Петр"])
    hypergraph.add_edges_from([("Иван", "Аня"), ("Иван", "Петр"), ("Аня", "Петр")])
    
    # Проверка, что метод visualize_hypergraph выполняется без ошибок
    try:
        hypergraph_builder.visualize_hypergraph(hypergraph)
    except Exception as e:
        pytest.fail(f"Ошибка при визуализации гиперграфа: {e}")

def test_save_and_load_hypergraph(hypergraph_builder, tmp_path):
    """Тестирование сохранения и загрузки гиперграфа."""
    hypergraph = nx.Graph()
    hypergraph.add_nodes_from(["Иван", "Аня", "Петр"])
    hypergraph.add_edges_from([("Иван", "Аня"), ("Иван", "Петр"), ("Аня", "Петр")])
    
    # Сохранение гиперграфа в GML
    file_path = tmp_path / "hypergraph.gml"
    hypergraph_builder.save_hypergraph(hypergraph, file_path, format="gml")
    
    # Загрузка гиперграфа из файла
    loaded_hypergraph = hypergraph_builder.load_hypergraph(file_path, format="gml")
    
    # Проверка, что гиперграф был правильно загружен
    assert len(loaded_hypergraph.nodes) == 3
    assert len(loaded_hypergraph.edges) == 3

@pytest.mark.parametrize("format", ["gml", "graphml"])
def test_save_hypergraph_invalid_format(hypergraph_builder, tmp_path, format):
    """Тестирование сохранения гиперграфа с некорректным форматом."""
    hypergraph = nx.Graph()
    hypergraph.add_nodes_from(["Иван", "Аня", "Петр"])
    hypergraph.add_edges_from([("Иван", "Аня"), ("Иван", "Петр"), ("Аня", "Петр")])
    
    with pytest.raises(ValueError):
        hypergraph_builder.save_hypergraph(hypergraph, tmp_path / "hypergraph.invalid", format="invalid_format")

def test_describe_hypergraph(hypergraph_builder):
    """Тестирование метода describe_hypergraph."""
    hypergraph = nx.Graph()
    hypergraph.add_nodes_from(["Иван", "Аня", "Петр"])
    hypergraph.add_edges_from([("Иван", "Аня"), ("Иван", "Петр"), ("Аня", "Петр")])
    
    description = hypergraph_builder.describe_hypergraph(hypergraph)
    
    # Проверка структуры описания гиперграфа
    assert description == {
        "Иван": ["Аня", "Петр"],
        "Аня": ["Иван", "Петр"],
        "Петр": ["Иван", "Аня"]
    }

