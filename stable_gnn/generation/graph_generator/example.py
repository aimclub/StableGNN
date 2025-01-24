# Импортируем класс
from llm_client import LLMClient  # Замените на правильный путь до вашего модуля

# Создаём экземпляр клиента
client = LLMClient(model="qwen:32b")

# Текст для обработки
text = """
Иван и Аня пошли в парк, а Петр присоединился к ним позже. В парке они встретили друзей. Все вместе они пошли в кафе.
"""

# Генерация графа
graph = client.generate_graph_description(text)
print("Сгенерированный граф:")
print(graph)

# Генерация гиперграфа
hypergraph = client.generate_hypergraph_description(text)
print("Сгенерированный гиперграф:")
print(hypergraph)
