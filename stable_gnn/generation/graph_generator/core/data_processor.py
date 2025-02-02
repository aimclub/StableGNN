import re
from typing import Dict, List


class DataProcessor:
    def __init__(self):
        """Инициализация процессора данных."""
        pass

    def clean_text(self, text: str) -> str:
        """
        Очистка текста от лишних символов и приведение к нижнему регистру.

        :param text: Исходный текст.
        :return: Очищенный текст.
        """
        # Удаление специальных символов и чисел
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)
        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление лишних пробелов
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_entities(self, text: str) -> List[str]:
        """
        Извлечение сущностей из текста. В данном примере сущностями считаются имена собственные на русском языке.

        :param text: Оригинальный текст.
        :return: Список сущностей.
        """
        # Изменение: работаем с оригинальным текстом, а не с очищенным
        entities = re.findall(r"\b[А-ЯЁ][а-яё]+\b", text)
        return entities

    def summarize_text(self, text: str) -> str:
        """
        Краткое резюме текста.

        :param text: Исходный текст.
        :return: Резюме текста.
        """
        # Простой пример: возвращаем первые 100 символов
        return text[:100] + "..." if len(text) > 100 else text

    def tokenize_text(self, text: str) -> List[str]:
        """
        Токенизация текста на слова.

        :param text: Очищенный текст.
        :return: Список токенов.
        """
        tokens = text.split()
        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Удаление стоп-слов из списка токенов.

        :param tokens: Список токенов.
        :return: Список токенов без стоп-слов.
        """
        stopwords = self._get_stopwords()
        filtered_tokens = [token for token in tokens if token not in stopwords]
        return filtered_tokens

    def _get_stopwords(self) -> List[str]:
        """
        Возвращает список стоп-слов.

        :return: Список стоп-слов.
        """
        # Добавлено 'это'
        return [
            "и",
            "в",
            "во",
            "не",
            "что",
            "он",
            "на",
            "я",
            "с",
            "со",
            "как",
            "а",
            "то",
            "все",
            "она",
            "так",
            "его",
            "но",
            "да",
            "ты",
            "к",
            "у",
            "же",
            "вы",
            "за",
            "бы",
            "по",
            "только",
            "ее",
            "мне",
            "было",
            "вот",
            "от",
            "меня",
            "еще",
            "нет",
            "о",
            "из",
            "ему",
            "теперь",
            "когда",
            "даже",
            "ну",
            "вдруг",
            "ли",
            "если",
            "уже",
            "или",
            "ни",
            "быть",
            "был",
            "него",
            "до",
            "вас",
            "нибудь",
            "опять",
            "уж",
            "вам",
            "ведь",
            "там",
            "потом",
            "себя",
            "ничего",
            "ей",
            "может",
            "они",
            "тут",
            "где",
            "есть",
            "надо",
            "ней",
            "для",
            "мы",
            "тебя",
            "их",
            "чем",
            "была",
            "сам",
            "чтоб",
            "без",
            "будто",
            "чего",
            "раз",
            "тоже",
            "себе",
            "под",
            "будет",
            "это",  # Добавлено
        ]
