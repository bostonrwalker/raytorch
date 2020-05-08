from raytorch.util import auto_str, clamp
import math


@auto_str
class RGB:

    def __init__(self, r: float, g: float, b: float):
        self.r = r
        self.g = g
        self.b = b

    def asarray(self):
        return [self.r, self.g, self.b]

    def astuple(self):
        return self.r, self.g, self.b

    def asinttuple(self):
        return clamp(int(math.floor(256 * self.r)), 0, 255), clamp(int(math.floor(256 * self.g)), 0, 255),\
               clamp(int(math.floor(256 * self.b)), 0, 255)

    def __add__(self, other):
        if isinstance(other, RGB):
            return RGB(self.r + other.r, self.g + other.g, self.b + other.b)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return RGB(self.r * other, self.g * other, self.b * other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return RGB(self.r * other, self.g * other, self.b * other)
        return NotImplemented


Black = RGB(.0, .0, .0)
