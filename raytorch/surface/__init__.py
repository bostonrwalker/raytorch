from raytorch.core import Matrix
from abc import ABC, abstractmethod


class Surface(ABC):
    """
    A class representing a 2-D manifold embedded in 3-D, to be rendered
    """

    def __init__(self, texture: int):
        self.texture = texture

    def __rmatmul__(self, other):
        if isinstance(other, Matrix):
            return self._apply_matrix(other)
        return NotImplemented

    @abstractmethod
    def _apply_matrix(self, matrix: Matrix):
        pass
