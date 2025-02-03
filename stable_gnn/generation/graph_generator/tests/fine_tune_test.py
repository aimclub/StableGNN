import json
import os
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from stable_gnn.generation.core.fine_tune_client import FineTuneClient
from stable_gnn.generation.graph_generator.core.data_processor import DataProcessor


# Создание MockTokenizer с необходимыми атрибутами
class MockTokenizer:
    def __init__(self):
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token_id = 50256
        self.eos_token_id = 50256
        self.vocab_size = 50257

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, truncation=True, padding="max_length", max_length=128):
        # Возвращает фиктивные данные, соответствующие токенизатору
        return {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }


@pytest.fixture
def data_processor_mock():
    """Фикстура для мокированного DataProcessor."""
    data_processor = MagicMock(spec=DataProcessor)
    data_processor.clean_text.return_value = "привет мир это тестовый текст"
    data_processor.extract_entities.return_value = ["привет", "мир", "тестовый", "текст"]
    return data_processor


@pytest.fixture
def fine_tune_client(data_processor_mock, tmp_path):
    """Фикстура для инициализации FineTuneClient с мокированным DataProcessor и замокированной загрузкой модели."""
    with patch(
        "graph_generator.core.fine_tune_client.AutoModelForCausalLM.from_pretrained"
    ) as mock_from_pretrained_model, patch(
        "graph_generator.core.fine_tune_client.AutoTokenizer.from_pretrained"
    ) as mock_from_pretrained_tokenizer:
        # Мокируем возвращаемый объект модели
        mock_model = MagicMock()
        mock_from_pretrained_model.return_value = mock_model

        # Мокируем метод resize_token_embeddings
        mock_model.resize_token_embeddings = MagicMock()

        # Мокируем возвращаемый объект токенизатора
        mock_tokenizer = MockTokenizer()
        mock_from_pretrained_tokenizer.return_value = mock_tokenizer

        output_dir = tmp_path / "fine_tuned_model"
        client = FineTuneClient(
            model_name="gpt2",
            tokenizer_name="gpt2",
            output_dir=str(output_dir),
            data_processor=data_processor_mock,
        )
    return client


def test_prepare_training_data_content(fine_tune_client, data_processor_mock):
    """Дополнительный тест для проверки содержимого подготовленных данных."""
    raw_data = "Привет, мир! Это тестовый текст №1."
    prepared_data = fine_tune_client.prepare_training_data(raw_data)
    dataset = prepared_data
    entry = json.loads(dataset["text"][0])

    assert entry["prompt"] == "привет мир это тестовый текст"
    assert json.loads(entry["completion"]) == {"entities": ["привет", "мир", "тестовый", "текст"]}


def test_prepare_training_data(fine_tune_client, data_processor_mock):
    """Тестирование метода prepare_training_data."""
    raw_data = "Привет, мир! Это тестовый текст №1."
    dataset = fine_tune_client.prepare_training_data(raw_data)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 1
    entry = json.loads(dataset["text"][0])
    assert entry["prompt"] == "привет мир это тестовый текст"
    assert json.loads(entry["completion"]) == {"entities": ["привет", "мир", "тестовый", "текст"]}


def test_upload_training_data(fine_tune_client, data_processor_mock):
    """Тестирование метода upload_training_data."""
    # Создание фиктивного Dataset
    raw_data = "Привет, мир! Это тестовый текст №1."
    dataset = fine_tune_client.prepare_training_data(raw_data)

    # Патчим save_to_disk, чтобы избежать реальных операций с диском
    with patch("graph_generator.core.fine_tune_client.Dataset.save_to_disk") as mock_save_to_disk:
        # Вызов метода upload_training_data
        fine_tune_client.upload_training_data(dataset)

        # Проверка, что save_to_disk был вызван один раз с правильным путём
        tokenized_data_dir = os.path.join(fine_tune_client.output_dir, "tokenized_data")
        mock_save_to_disk.assert_called_once_with(tokenized_data_dir)


def test_check_fine_tuning_status(fine_tune_client):
    """Тестирование метода check_fine_tuning_status."""
    status = fine_tune_client.check_fine_tuning_status()
    assert status == "completed"


def test_download_fine_tuned_model_success(fine_tune_client, data_processor_mock, tmp_path):
    """Тестирование метода download_fine_tuned_model при успешном сохранении."""
    with patch("graph_generator.core.fine_tune_client.shutil.copytree") as mock_copytree:
        # Создание фиктивной тонко настроенной модели
        model_dir = os.path.join(fine_tune_client.output_dir, "fine_tuned_model")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "pytorch_model.bin"), "wb") as f:
            f.write(b"dummy_model_data")

        # Путь для сохранения
        save_path = tmp_path / "saved_fine_tuned_model"

        # Вызов метода download_fine_tuned_model
        success = fine_tune_client.download_fine_tuned_model(str(save_path))

        # Проверка
        assert success is True
        mock_copytree.assert_called_once_with(model_dir, str(save_path), dirs_exist_ok=True)


def test_download_fine_tuned_model_failure(fine_tune_client, data_processor_mock, tmp_path):
    """Тестирование метода download_fine_tuned_model при ошибке сохранения."""
    with patch(
        "graph_generator.core.fine_tune_client.shutil.copytree", side_effect=Exception("Copy failed")
    ) as mock_copytree:
        # Создание фиктивной тонко настроенной модели
        model_dir = os.path.join(fine_tune_client.output_dir, "fine_tuned_model")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "pytorch_model.bin"), "wb") as f:
            f.write(b"dummy_model_data")

        # Путь для сохранения
        save_path = tmp_path / "saved_fine_tuned_model"

        # Вызов метода
        success = fine_tune_client.download_fine_tuned_model(str(save_path))

        # Проверка
        assert success is False
        mock_copytree.assert_called_once_with(model_dir, str(save_path), dirs_exist_ok=True)
