from raytorch.surface import Surface
from raytorch.core import Vector, Matrix
from raytorch.util import auto_str


@auto_str
class Parallelogram(Surface):

    def __init__(self, pos: Vector, bottom_edge: Vector, left_edge: Vector, texture: int):
        self.pos = pos
        self.bottom_edge = bottom_edge
        self.left_edge = left_edge
        super().__init__(texture)

    def normal_vector(self):
        n = self.bottom_edge.cross(self.left_edge)
        return n / n.norm()

    def _apply_matrix(self, matrix: Matrix):
        return Parallelogram(matrix @ self.pos,
                             matrix @ (self.pos + self.bottom_edge) - matrix @ self.pos,
                             matrix @ (self.pos + self.left_edge) - matrix @ self.pos,
                             texture=self.texture)
