from raytorch.util import auto_str
from raytorch.core import Vector, Matrix


@auto_str
class Camera:

    def __init__(self, pos: Vector, dir: Vector, up: Vector, right: Vector):
        self.pos = pos
        self.dir = dir
        self.up = up
        self.right = right

    def look_at(self) -> Matrix:
        a = Matrix(self.right.x, self.right.y, self.right.z, 0.,
                   self.up.x, self.up.y, self.up.z, 0.,
                   self.dir.x, self.dir.y, self.dir.z, 0.,
                   0., 0., 0., 1.)
        b = Matrix(1., 0., 0., -self.pos.x,
                   0., 1., 0., -self.pos.y,
                   0., 0., 1., -self.pos.z,
                   0., 0., 0., 1.)
        return a @ b

    @staticmethod
    def upright_camera(pos: Vector, target: Vector):
        """
        Create an upright camera (i.e. up Vector maximally aligned with +y axis)
        Undefined if target is directly above or below position
        https://learnopengl.com/Getting-started/Camera
        """
        dir = pos - target
        dir /= dir.norm()
        right = Vector(0, 1, 0).cross(dir)
        right /= right.norm()
        up = dir.cross(right)
        up /= up.norm()
        return Camera(pos, dir, up, right)
