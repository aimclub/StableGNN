import dataclasses

import vk

from gns.crawlers.vk.app.config.config import Config, get_config


@dataclasses.dataclass
class VKAPI:
    api = None
    _config: Config

    def connect(self):
        self.api = vk.API(access_token=self._config.access_key)


def get_vk_api_fabric() -> VKAPI:
    return VKAPI(_config=get_config())
