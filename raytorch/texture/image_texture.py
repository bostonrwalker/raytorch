from raytorch.texture import Texture
from raytorch.core import RGB
import math
from PIL import Image
from os.path import abspath


class ImageTexture(Texture):

    def __init__(self, path: str, use_tensors=False):
        self.path = path
        self.pixels = None
        self.dim = None
        self.use_tensors = use_tensors

        self.ambient = None

    def load(self):
        if self.path is not None:
            self.ambient = Image.open(abspath(self.path), 'r')

    def texel_at(self, u, v):
        """
        Return pixel colour
        :param u: left to right in range [0, 1)
        :param v: bottom to top in range [0, 1)
        """
        if u < 0. or u >= 1. or v < 0. or v >= 1.:
            raise ValueError('Inputs out of bounds: {x}, {y}')
        px = int(math.floor(u * self.dim[0]))
        py = int(math.floor((1 - v) * self.dim[1]))
        return RGB(*(p / 255. for p in self.pixels[py * self.dim[0] + px]))

    def __str__(self):
        return(f'ImageTexture{{path={str(self.path)},'
               f'pixels={"[(int,)]" if self.pixels else "None"},'
               f'dim={str(self.dim)}}}')

    def __repr__(self):
        return(f'ImageTexture{{path={repr(self.path)},'
               f'pixels={"[(int,)]" if self.pixels else "None"},'
               f'dim={repr(self.dim)}}}')
