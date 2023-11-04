import dataclasses

from gns.crawlers.vk.app.api.vkapi import VKAPI, get_vk_api_fabric
from gns.crawlers.vk.app.config.config import Config, get_config


@dataclasses.dataclass
class Group:
    _vk: VKAPI
    _config: Config

    def __post_init__(self):
        self._vk.connect()

    def get_members(self, group_id: str):
        first = self._vk.api.groups.getMembers(group_id=group_id, v=self._config.v)
        data = first["items"]
        count = first["count"] // self._config.offset_modifier

        for i in range(1, count + 1):
            data = (
                data
                + self._vk.api.groups.getMembers(
                    group_id=group_id,
                    v=self._config.v,
                    offset=i * self._config.offset_modifier,
                )["items"]
            )

        for d in data:
            yield d


def get_group_api_fabric() -> Group:
    return Group(_vk=get_vk_api_fabric(), _config=get_config())
