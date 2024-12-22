import pytest
from graph_generator.core.data_processor import DataProcessor

@pytest.fixture
def data_processor():
    """Фикстура для инициализации DataProcessor."""
    return DataProcessor()

def test_clean_text(data_processor):
    """Тестирование метода clean_text."""
    input_text = "Привет, мир! Это тестовый текст №1."
    expected_output = "привет мир это тестовый текст"
    assert data_processor.clean_text(input_text) == expected_output

def test_extract_entities(data_processor):
    """Тестирование метода extract_entities."""
    input_text = "Иван и Аня пошли в парк, а Петр присоединился к ним позже."
    expected_entities = ["Иван", "Аня", "Петр"]
    assert data_processor.extract_entities(input_text) == expected_entities

def test_summarize_text(data_processor):
    """Тестирование метода summarize_text."""
    input_text = "Это короткий текст."
    expected_summary = "Это короткий текст."
    assert data_processor.summarize_text(input_text) == expected_summary

    long_text = "a" * 150
    expected_summary_long = "a" * 100 + "..."
    assert data_processor.summarize_text(long_text) == expected_summary_long

def test_tokenize_text(data_processor):
    """Тестирование метода tokenize_text."""
    input_text = "это тестовый текст для токенизации"
    expected_tokens = ["это", "тестовый", "текст", "для", "токенизации"]
    assert data_processor.tokenize_text(input_text) == expected_tokens

def test_remove_stopwords(data_processor):
    """Тестирование метода remove_stopwords."""
    tokens = ["это", "тестовый", "текст", "для", "токенизации"]
    expected_filtered = ["тестовый", "текст", "токенизации"]
    assert data_processor.remove_stopwords(tokens) == expected_filtered
