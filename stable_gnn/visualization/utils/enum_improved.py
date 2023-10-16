from enum import Enum, EnumMeta


class EnumMetaClass(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class EnumImproved(Enum, metaclass=EnumMetaClass):

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
