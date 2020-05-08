from raytorch.surface import Surface
from raytorch.core import Vector, Matrix
from raytorch.util import auto_str

@auto_str
class Ellipsoid(Surface):

    def __init__(self, pos: Vector, first_axis: Vector, second_axis: Vector, third_axis: Vector, texture: int,
                 inverted=False):
        self.pos = pos
        self.first_axis = first_axis
        self.second_axis = second_axis
        self.third_axis = third_axis
        self.inverted = inverted
        super().__init__(texture)

    def _apply_matrix(self, matrix: Matrix):
        new_pos = matrix @ self.pos
        return Ellipsoid(new_pos,
                         matrix @ (self.pos + self.first_axis) - new_pos,
                         matrix @ (self.pos + self.second_axis) - new_pos,
                         matrix @ (self.pos + self.third_axis) - new_pos,
                         texture=self.texture, inverted=self.inverted)
