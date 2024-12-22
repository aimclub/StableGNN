import ollama
import json
import time
import logging
from graph_generator.core.data_processor import DataProcessor

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model: str = "mistral:7b", data_processor: DataProcessor = None):
        """
        Инициализация клиента для работы с Ollama.
        :param model: Модель, которая будет использоваться.
        :param data_processor: Экземпляр DataProcessor для предварительной обработки текста.
        """
        self.model = model
        self.data_processor = data_processor or DataProcessor()
    
    def generate_graph_description(self, text: str) -> dict:
        """
        Генерация описания графа из текста.
        :param text: Входной текст для генерации графа.
        :return: Структура данных, описывающая граф.
        """
        # Предварительная обработка текста
        cleaned_text = self.data_processor.clean_text(text)
        system_prompt = """
        Ты должен проанализировать предоставленный текст и создать граф, где каждый объект (персонаж, место, событие и т. д.) будет представлен как узел, а связи между ними — как рёбра. Тебе нужно вывести структуру графа, где:
        - Узлы — это элементы текста (например, персонажи, места, события).
        - Рёбра — это связи между этими элементами, которые можно извлечь из контекста текста (например, взаимодействия персонажей, географические связи, причинно-следственные отношения).
    
        Пожалуйста, соблюдай структуру вывода:
        {
            "node_1": ["node_2", "node_3"],
            "node_2": ["node_1", "node_3"],
            ...
        }
    
        Узлы и рёбра должны быть названы в соответствии с контекстом текста. Ответ должен быть строго в формате JSON.
        """
        
        return self._generate_description(cleaned_text, system_prompt)
    
    def generate_hypergraph_description(self, text: str) -> dict:
        """
        Генерация описания гиперграфа из текста.
        :param text: Входной текст для генерации гиперграфа.
        :return: Структура данных, описывающая гиперграф.
        """
        # Предварительная обработка текста
        cleaned_text = self.data_processor.clean_text(text)
        system_prompt = """
        Ты должен проанализировать предоставленный текст и создать гиперграф, где каждый объект (персонаж, место, событие и т. д.) будет представлен как узел, а связи между ними — как гиперрёбра. Тебе нужно вывести структуру гиперграфа, где:
        - Узлы — это элементы текста (например, персонажи, места, события).
        - Гиперрёбра — это связи между несколькими элементами, которые можно извлечь из контекста текста (например, группы персонажей, события с множественными участниками, совместные действия).
    
        Пожалуйста, соблюдай структуру вывода:
        {
            "hyperedge_1": ["node_1", "node_2", "node_3"],
            "hyperedge_2": ["node_1", "node_4"],
            ...
        }
    
        Узлы и гиперрёбра должны быть названы в соответствии с контекстом текста. Ответ должен быть строго в формате JSON.
        """
        
        return self._generate_description(cleaned_text, system_prompt)
    
    def _generate_description(self, text: str, system_prompt: str) -> dict:
        """
        Генерация описания (граф или гиперграф) с повторной попыткой, если результат некорректный.
        :param text: Входной текст.
        :param system_prompt: Системный промпт для генерации.
        :return: Структура данных, описывающая граф или гиперграф.
        """
        retries = 3  # Количество попыток перегенерации
        while retries > 0:
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ]
                )
                content = response.get('message', {}).get('content', '{}')
                graph_description = json.loads(content)
                
                # Проверка, что ответ — словарь
                if not isinstance(graph_description, dict):
                    raise ValueError("Ответ должен быть в виде словаря (dict).")
                
                # Проверка, что все значения — непустые списки
                for key, value in graph_description.items():
                    if not isinstance(value, list) or not value:
                        raise ValueError(f"Структура должна содержать непустые списки для ключа '{key}'.")
                
                return graph_description
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Ошибка при обработке ответа: {e}")
                retries -= 1
                logger.info(f"Повторная попытка {4 - retries}/3 через 2 секунды...")
                time.sleep(2)  # Задержка перед повторной попыткой
            except Exception as e:
                logger.error(f"Произошла непредвиденная ошибка: {e}")
                retries -= 1
                logger.info(f"Повторная попытка {4 - retries}/3 через 2 секунды...")
                time.sleep(2)
        
        # Если все попытки исчерпаны, вернуть пустой словарь
        logger.error("Все попытки генерации описания завершились неудачно.")
        return {}
    
    def parse_response(self, response: dict) -> dict:
        """
        Парсинг ответа от Ollama в структуру данных для графа или гиперграфа.
        :param response: Ответ от Ollama.
        :return: Структура {node: [edges]} или {hyperedge: [nodes]}.
        """
        try:
            graph_description = json.loads(response.get('message', {}).get('content', '{}'))
            
            # Проверка корректности структуры данных
            if not isinstance(graph_description, dict):
                raise ValueError("Ответ должен быть в виде словаря (dict).")
            
            # Преобразуем и возвращаем граф или гиперграф в удобном для использования формате
            return graph_description
        except Exception as e:
            logger.error(f"Ошибка при парсинге ответа: {e}")
            return {}
