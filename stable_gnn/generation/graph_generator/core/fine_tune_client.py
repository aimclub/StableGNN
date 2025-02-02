import json
import os

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from stable_gnn.generation.graph_generator.core.base_fine_tune_client import BaseFineTuneClient
from stable_gnn.generation.graph_generator.core.data_processor import DataProcessor


class FineTuneClient(BaseFineTuneClient):
    def __init__(
        self,
        model_name: str = "gpt2",
        tokenizer_name: str = None,
        output_dir: str = "./fine_tuned_model",
        data_processor: DataProcessor = None,
    ):
        """
        Инициализация клиента для тонкой настройки модели с использованием Transformers.

        :param model_name: Название предобученной модели из Hugging Face.
        :param tokenizer_name: Название токенизатора. Если None, будет использовано имя модели.
        :param output_dir: Путь для сохранения тонко настроенной модели.
        :param data_processor: Экземпляр DataProcessor для предварительной обработки данных.
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.output_dir = output_dir
        self.data_processor = data_processor or DataProcessor()

        # Загрузка токенизатора и модели
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Создание директории для сохранения модели
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_training_data(self, raw_data: str) -> Dataset:
        """
        Подготовка и очистка данных для обучения.

        :param raw_data: Сырые данные для обучения.
        :return: Объект Dataset для обучения.
        """
        cleaned_text = self.data_processor.clean_text(raw_data)
        entities = self.data_processor.extract_entities(raw_data)

        # Формирование примера обучения
        training_entry = {
            "prompt": cleaned_text,
            "completion": json.dumps({"entities": entities}),
        }

        # Создание объекта Dataset
        dataset = Dataset.from_dict({"text": [json.dumps(training_entry)]})
        return dataset

    def tokenize_function(self, examples):
        """Токенизация примеров."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    def upload_training_data(self, training_data: Dataset) -> None:
        """
        Подготовка и сохранение данных для тонкой настройки.

        :param training_data: Объект Dataset для обучения.
        """
        # Токенизация данных
        tokenized_datasets = training_data.map(self.tokenize_function, batched=True)

        # Сохранение токенизированных данных в формате JSONL
        tokenized_datasets.save_to_disk(os.path.join(self.output_dir, "tokenized_data"))

    def start_fine_tuning(self, job_name: str = "fine_tune_job") -> None:
        """
        Запуск процесса тонкой настройки модели.

        :param job_name: Имя задания для тонкой настройки.
        """
        # Загрузка токенизированных данных
        tokenized_datasets = load_dataset("json", data_dir=os.path.join(self.output_dir, "tokenized_data"))

        # Настройки обучения
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            evaluation_strategy="no",
            learning_rate=5e-5,
            weight_decay=0.01,
            push_to_hub=False,
        )

        # Инициализация Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
        )

        # Запуск обучения
        trainer.train()

        # Сохранение тонко настроенной модели
        trainer.save_model(os.path.join(self.output_dir, "fine_tuned_model"))

    def check_fine_tuning_status(self) -> str:
        """
        Проверка статуса задания тонкой настройки.
        В данном случае, поскольку процесс обучения локальный, статус всегда 'completed'.

        :return: Статус задания.
        """
        return "completed"

    def download_fine_tuned_model(self, save_path: str) -> bool:
        """
        Копирование тонко настроенной модели в указанное место.

        :param save_path: Путь для сохранения модели.
        :return: True, если сохранение прошло успешно, иначе False.
        """
        try:
            model_dir = os.path.join(self.output_dir, "fine_tuned_model")
            if not os.path.exists(model_dir):
                raise FileNotFoundError("Тонко настроенная модель не найдена.")

            # Копирование модели
            import shutil

            shutil.copytree(model_dir, save_path, dirs_exist_ok=True)
            return True
        except Exception as e:
            print(f"Ошибка при сохранении тонко настроенной модели: {e}")
            return False
