import json
from unittest.mock import patch

import pytest

from stable_gnn.generation.graph_generator.core.llm_client import LLMClient


@pytest.fixture
def llm_client():
    """Фикстура для инициализации клиента LLM."""
    return LLMClient(model="qwen:32b")


@pytest.fixture
def mock_graph_response():
    """Фикстура для мокирования успешного ответа от Ollama для графа."""
    return {
        "message": {
            "content": json.dumps(
                {"Алексей": ["Иван", "Анна"], "Иван": ["Алексей", "Анна"], "Анна": ["Иван", "Алексей"]}
            )
        }
    }


@pytest.fixture
def mock_hypergraph_response():
    """Фикстура для мокирования успешного ответа от Ollama для гиперграфа."""
    return {"message": {"content": json.dumps({"hyperedge_1": ["Алексей", "Иван", "Анна"]})}}


@pytest.fixture
def mock_invalid_response():
    """Фикстура для мокирования некорректного ответа от Ollama."""
    return {"message": {"content": '{"invalid": "structure"}'}}


@pytest.fixture
def mock_empty_response():
    """Фикстура для мокирования пустого ответа от Ollama."""
    return {"message": {"content": json.dumps({})}}


@patch("ollama.chat")
def test_generate_graph_description(mock_chat, llm_client, mock_graph_response):
    """Тестирование успешной генерации графа с корректным ответом от Ollama."""
    mock_chat.return_value = mock_graph_response
    text = "Алексей встретился с Иваном в парке. Они поехали на концерт, а потом встретили Анну."

    description = llm_client.generate_graph_description(text)
    expected = {"Алексей": ["Иван", "Анна"], "Иван": ["Алексей", "Анна"], "Анна": ["Иван", "Алексей"]}
    assert description == expected


@patch("ollama.chat")
def test_generate_hypergraph_description(mock_chat, llm_client, mock_hypergraph_response):
    """Тестирование успешной генерации гиперграфа с корректным ответом от Ollama."""
    mock_chat.return_value = mock_hypergraph_response
    text = "Алексей, Иван и Анна поехали на концерт, а после этого все встретились в кафе."

    description = llm_client.generate_hypergraph_description(text)
    expected = {"hyperedge_1": ["Алексей", "Иван", "Анна"]}
    assert description == expected


@patch("ollama.chat")
def test_generate_graph_description_invalid_response(mock_chat, llm_client, mock_invalid_response):
    """Тестирование обработки некорректного ответа от Ollama."""
    mock_chat.return_value = mock_invalid_response
    text = "Текст с ошибочной структурой для графа."

    description = llm_client.generate_graph_description(text)
    assert description == {}


@patch("ollama.chat")
def test_generate_hypergraph_description_invalid_response(mock_chat, llm_client, mock_invalid_response):
    """Тестирование обработки некорректного ответа для гиперграфа от Ollama."""
    mock_chat.return_value = mock_invalid_response
    text = "Текст с ошибочной гиперграфической структурой."

    description = llm_client.generate_hypergraph_description(text)
    assert description == {}


@patch("ollama.chat")
def test_generate_graph_description_empty_response(mock_chat, llm_client, mock_empty_response):
    """Тестирование обработки пустого ответа от Ollama."""
    mock_chat.return_value = mock_empty_response
    text = "Текст, не содержащий связей для графа."

    description = llm_client.generate_graph_description(text)
    assert description == {}


@patch("ollama.chat")
def test_generate_hypergraph_description_empty_response(mock_chat, llm_client, mock_empty_response):
    """Тестирование обработки пустого ответа для гиперграфа от Ollama."""
    mock_chat.return_value = mock_empty_response
    text = "Текст, не содержащий гиперсвязей для гиперграфа."

    description = llm_client.generate_hypergraph_description(text)
    assert description == {}


@patch("ollama.chat")
def test_retry_on_invalid_response(mock_chat, llm_client, mock_invalid_response, mock_graph_response):
    """Тестирование логики повторных попыток при получении некорректного ответа."""
    mock_chat.side_effect = [mock_invalid_response, mock_invalid_response, mock_graph_response]
    text = "Текст, который должен вернуть ошибочную структуру, но затем правильную."

    description = llm_client.generate_graph_description(text)
    expected = {"Алексей": ["Иван", "Анна"], "Иван": ["Алексей", "Анна"], "Анна": ["Иван", "Алексей"]}
    assert description == expected
