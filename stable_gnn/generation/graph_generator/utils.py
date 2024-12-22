import json

def save_to_file(data, filename: str):
    """
    Сохранение данных в файл в формате JSON.
    :param data: Данные для сохранения.
    :param filename: Путь до файла.
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)  # Добавил indent для форматирования JSON
        print(f"Data saved into file {filename}.")
    except (IOError, json.JSONDecodeError) as e:
        print(f"File save error {filename}: {e}")
        raise

def load_from_file(filename: str):
    """
    Загрузка данных из файла в формате JSON.
    :param filename: Путь до файла.
    :return: Загруженные данные.
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Data parsing error {filename}. Maybe file corrupted.")
        return None
    except IOError as e:
        print(f"File open error {filename}: {e}")
        return None
