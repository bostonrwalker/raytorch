from abc import ABC, abstractmethod


class Texture(ABC):
    """
    A class to represent a texture, i.e. a mapping of surface space and incident angle to shader properties
    """

    @abstractmethod
    def load(self):
        """
        Load texture into memory, if necessary
        """
        pass
