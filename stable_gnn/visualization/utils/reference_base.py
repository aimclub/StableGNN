

class ReferenceBase:

    @property
    def values(self):
        properties = []
        for attribute_name, attribute_value in self.__dict__.items():
            if not callable(attribute_value):
                properties.append(attribute_value)
        return properties
