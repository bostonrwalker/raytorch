from raytorch.texture.fresnel_texture import FresnelTexture
from raytorch.core import RGB
import math


class Metal(FresnelTexture):

    def __init__(self, surface_col=None, normal_map_path=None):

        if surface_col is None:
            surface_col = RGB(.3, .3, .3)

        def reflectivity(theta):
            pr = 0.4 + 0.6 * (theta / (math.pi / 2.)) ** 4
            pt = 0.
            pa = 0.6 * (1. - (theta / (math.pi / 2.)) ** 4)
            return pr, pt, pa

        super().__init__(surface_col=surface_col, reflectivity_func=reflectivity, normal_map_path=normal_map_path)
