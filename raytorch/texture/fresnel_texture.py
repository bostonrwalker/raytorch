from typing import Callable, Union

from pathlib import Path

from PIL import Image

from raytorch.texture import Texture
from raytorch.core import RGB
from raytorch.util import auto_str


@auto_str
class FresnelTexture(Texture):

    def __init__(self, surface_col: RGB, reflectivity_func: Callable, refraction_ind: float = 1.,
                 normal_map_path: Union[Path, str, None] = None):
        """
        :param surface_col: The ambient colour of the surface
        :param reflectivity_func: A function of theta (angle of reflection vs. the normal). Return a tuple containing:
                    pr: (float) Proportion of light reflected
                    pt: (float) Proportion of light transmitted
                    pa: (float) Proportion of ambient light
        :param refraction_ind: Index of refraction
        :param normal_map_path: Path to Dot3 normal map file
        """
        self.surface_col = surface_col
        self.reflectivity_func = reflectivity_func
        self.refraction_ind = refraction_ind
        self.normal_map_path = normal_map_path

        self.normal_map = None

    def load(self):
        if self.normal_map_path is not None:
            self.normal_map = Image.open(self.normal_map_path, 'r')
