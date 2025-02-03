import json


def save_to_file(data, filename: str):
    """
    Сохранение данных в файл в формате JSON.
    :param data: Данные для сохранения.
    :param filename: Путь до файла.
    """
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)  # Добавил indent для форматирования JSON
        print(f"Данные успешно сохранены в {filename}.")
    except (IOError, json.JSONDecodeError) as e:
        print(f"Ошибка при сохранении данных в файл {filename}: {e}")
        raise


def load_from_file(filename: str):
    """
    Загрузка данных из файла в формате JSON.
    :param filename: Путь до файла.
    :return: Загруженные данные.
    """
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
        return None
    except json.JSONDecodeError:
        print(f"Ошибка при разборе данных из файла {filename}. Возможно, файл поврежден.")
        return None
    except IOError as e:
        print(f"Ошибка при открытии файла {filename}: {e}")
        return None
