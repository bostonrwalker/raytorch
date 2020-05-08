from raytorch.surface.ellipsoid import Ellipsoid
from raytorch.core import Vector
from raytorch.util import auto_str


@auto_str
class Sphere(Ellipsoid):

    def __init__(self, pos: Vector, radius: float, texture: int, inverted=False):
        super().__init__(pos, Vector(radius, 0., 0.), Vector(0., radius, 0.), Vector(0., 0., radius),
                         texture=texture, inverted=inverted)
