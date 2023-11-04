import dataclasses
import os

from dotenv import load_dotenv

# Read envs
load_dotenv()


@dataclasses.dataclass
class Config:
    v: str = 5.131
    offset_modifier: int = 1000
    access_key = os.environ.get("VK_API_KEY", "")
    max_search_depth = 3


def get_config() -> Config:
    return Config()
