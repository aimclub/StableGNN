from functools import lru_cache


def cached(maxsize=32, typed=True):
    def decorator(func):
        cache = lru_cache(maxsize=maxsize, typed=typed)

        @cache
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
