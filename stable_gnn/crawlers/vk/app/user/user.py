import dataclasses
import time
import typing as t
from typing import List

from gns.crawlers.vk.app.api.vkapi import VKAPI, get_vk_api_fabric
from gns.crawlers.vk.app.config.config import Config, get_config
from gns.crawlers.vk.app.config.constants import Fields, all_fields

config = get_config()


@dataclasses.dataclass
class User:
    _vk: VKAPI
    _config: Config

    def __post_init__(self):
        self._vk.connect()

    @staticmethod
    def _list_to_param_string(params: List[t.Union[int, str]]):
        if not isinstance(params, list):
            params = [params]
        return ",".join(str(x) for x in params)

    def get_user_id_by_verbose_name(self, verbose_name):
        users = self._vk.api.users.get(user_ids=[verbose_name], v=self._config.v)
        if users:
            return users[0].get("id")

    def get_friends(self, user_id: str, fields: List[str] = [], name_case: str = "nom"):
        fields = self._list_to_param_string(fields) or ""
        if not fields:
            fields = self._list_to_param_string(all_fields())
        name_case = name_case or ""
        first = self._vk.api.friends.get(
            user_id=user_id,
            fields=fields,
            name_case=name_case,
            v=self._config.v,
        )

        data = first["items"]
        count = first["count"] // self._config.offset_modifier

        for i in range(1, count + 1):
            data = (
                data
                + self._vk.api.users.getFollowers(
                    user_id=user_id,
                    v=self._config.v,
                    name_case=name_case,
                    fields=fields,
                    offset=i * self._config.offset_modifier,
                )["items"]
            )

        return data

    def get_info(
        self,
        user_ids: t.Union[int, str, List[t.Union[int, str]]],
        fields: List[str] = [],
        name_case: str = "nom",
    ):
        """
        Get information about a user or about several users.

        Args:
            user_ids: comma-separated user IDs
            or their short names (screen_name). By default, the ID of the current
            the user. Comma-separated list of words, number of elements
            should be no more than 1000.
            fields: A list of additional profile fields that need to be returned.
            Possible values: Constants.Fields.
            name_case: the case for declension of the user's first and last name.
            Possible values: Constants.Name Case.

        Returns:

        """
        user_ids = self._list_to_param_string(user_ids)
        fields = self._list_to_param_string(fields) or ""
        if not fields:
            fields = self._list_to_param_string(all_fields())
        name_case = name_case or ""
        return self._vk.api.users.get(
            user_ids=user_ids, fields=fields, name_case=name_case, v=self._config.v
        )

    def get_followers_ids(self, user_id: int):
        followers = self.get_followers(user_id=user_id, fields=[Fields.timezone.value])
        if followers:
            return [f.get("id") for f in followers]

    def get_followers(self, user_id: int, fields=None, name_case: str = "nom"):
        """
        Returns a list of user IDs that are subscribers of the user.

        Args:
            user_id: User ID.
            fields: a list of additional profile fields that need to be returned.
            Possible values: Constants.Fields.
            name_case: The case for declension of the user's first and last name.
            Possible values: Constants.NameCase

        Returns:

        """
        if fields is None:
            fields = []
        fields = self._list_to_param_string(fields) or ""
        if not fields:
            fields = self._list_to_param_string(all_fields())
        name_case = name_case or ""
        first = self._vk.api.users.getFollowers(
            user_id=user_id,
            v=self._config.v,
            name_case=name_case,
            fields=fields,
        )
        data = first["items"]
        count = first["count"] // self._config.offset_modifier

        for i in range(1, count + 1):
            data = (
                data
                + self._vk.api.users.getFollowers(
                    user_id=user_id,
                    v=self._config.v,
                    name_case=name_case,
                    fields=fields,
                    offset=i * self._config.offset_modifier,
                )["items"]
            )

        return data

    def get_user_fields(self, user):
        return {
            "id": user["id"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
        }

    def get_groups(self, user_id: int, extended: bool = True, fields: List[str] = []):
        extended = 1 if extended else 0
        first = self._vk.api.groups.get(
            user_id=user_id, extended=extended, fields=fields, v=self._config.v
        )
        data = first["items"]
        count = first["count"] // self._config.offset_modifier

        for i in range(1, count + 1):
            time.sleep(0.5)
            data = (
                data
                + self._vk.api.groups.get(
                    user_id=user_id,
                    extended=extended,
                    fields=fields,
                    v=self._config.v,
                    offset=i * self._config.offset_modifier,
                )["items"]
            )

        return data

    def get_subscriptions(
        self, user_id: int, extended: bool = True, fields: List[str] = []
    ):
        """
        Возвращает список идентификаторов пользователей и публичных страниц, которые входят в список подписок пользователя.

        Args:
            user_id: ID of the user whose subscriptions you want to get.
            extended: 1 – returns a combined list containing group and user objects together. 0 – returns a list
            ids of groups and users separately (by default).
            fields: A list of additional fields for the user and group objects to be returned.

        Returns:

        """
        extended = 1 if extended else 0
        fields = self._list_to_param_string(fields) or ""
        if not fields:
            fields = self._list_to_param_string(all_fields())
        first = self._vk.api.users.getSubscriptions(
            user_id=user_id,
            extended=extended,
            fields=fields,
            v=self._config.v,
        )
        data = first["items"]
        count = first["count"] // self._config.offset_modifier

        for i in range(1, count + 1):
            data = (
                data
                + self._vk.api.users.getSubscriptions(
                    user_id=user_id,
                    extended=extended,
                    v=self._config.v,
                    fields=fields,
                    offset=i * self._config.offset_modifier,
                )["items"]
            )

        for d in data:
            yield d

    def search(
        self,
        q: str,
        sort: bool,
        offset: int,
        count: int,
        fields: List[str],
        city: int,
        country: int,
        hometown: str,
        university_country: int,
        university: int,
        university_year: int,
        university_faculty: int,
        university_chair: int,
        sex: int,
        status: int,
        age_from: int,
        age_to: int,
        birth_day: int,
        birth_month: int,
        birth_year: int,
        online: bool,
        has_photo: bool,
        school_country: int,
        school_city: int,
        school_class: int,
        school: int,
        school_year: int,
        religion: str,
        company: str,
        position: str,
        group_id: int,
        from_list: str,
    ):
        """
        Returns a list of users according to the specified search criteria.

        Args:
            q: The search query string.
            sort: Sorting the results. Possible values:
                1 — by registration date,
                0 — by popularity.
            offset: Offset relative to the first user found to select a specific subset.
            count: The number of users to return (maximum 1000).
            fields: List of additional profile fields.
                Possible values: Constants.Fields.
            city: ID of the city.
            country: Country ID.
            hometown: The name of the city is a string.
            university_country: ID of the country in which the users graduated from the university.
            university: ID of the university.
            university_year: The year of graduation.
            university_faculty: ID of the faculty.
            university_chair: ID of the department.
            sex: Sex.
            status: Marital status. Possible values:
                1 — not married (not married),
                2 — meets,
                3 — engaged,
                4 — married (married),
                5 — everything is complicated,
                6 — in active search,
                7 — in love (-a),
                8 — in a civil marriage.
            age_from: Age, from.
            age_to: Age, up to.
            birth_day: Birthday.
            birth_month: Month of birth.
            birth_year: Year of birth.
            online: Whether to take into account the "online" status. Possible values:
                1 — search for online users only,
                0 — search for all users.
            has_photo: Whether to take into account the presence of photos. Possible values:
                1 — search only for users with a photo,
                0 — search for all users.
            school_country: ID of the country where users graduated from school.
            school_city: ID of the city where users graduated from school.
            school_class: The letter of the class.
            school: ID of the school that the users graduated from.
            school_year: The year of graduation.
            religion: Religion.
            company: The name of the company where the users work.
            position: The title of the position.
            group_id: ID of the group to search among the users.
            from_list: The sections among which you need to search are listed separated by commas. Possible values:
                friends — search among friends,
                subscriptions — search among the user's friends and subscriptions.

        Returns:

        """
        raise NotImplementedError

    def build_followers_tree(
        self, user_id: int, max_depth: int = config.max_search_depth
    ):
        raise NotImplementedError


def get_user_api_fabric() -> User:
    return User(_vk=get_vk_api_fabric(), _config=get_config())
