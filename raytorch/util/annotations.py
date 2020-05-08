"""
Useful annotators
"""


def auto_str(cls):
    """
    Add implementation of __str__ and __repr__ functions to a class
    """

    def __str__(self):
        return type(self).__name__ + '{' + ','.join(f'{k}={str(v)}' for k, v, in vars(self).items()) + '}'

    def __repr__(self):
        return type(self).__qualname__ + '{' + ','.join(f'{k}={repr(v)}' for k, v, in vars(self).items()) + '}'

    cls.__str__ = __str__
    cls.__repr__ = __repr__
    return cls


def singleton(cls):
    """
    Add get_instance method
    """
    cls._instance = None

    def get_instance():
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    cls.get_instance = get_instance
    return cls
