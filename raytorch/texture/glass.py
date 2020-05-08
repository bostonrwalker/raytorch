from raytorch.texture.fresnel_texture import FresnelTexture
from raytorch.core import RGB
import math


class Glass(FresnelTexture):

    def __init__(self, surface_col=None, refraction_ind=1.517, inverted=False, normal_map_path=None):

        if surface_col is None:
            surface_col = RGB(.99, .99, .99)

        if inverted is True:
            refraction_ind = 1. / refraction_ind

        if refraction_ind < 1.:
            theta_crit = math.asin(refraction_ind)
        else:
            theta_crit = math.pi / 2

        def reflectivity(theta):
            pr = 0.1 + 0.86 * (theta / theta_crit) ** 4
            pt = 0.86 * (1. - (theta / theta_crit) ** 4)
            pa = 0.04
            return pr, pt, pa

        super().__init__(surface_col=surface_col, reflectivity_func=reflectivity, refraction_ind=refraction_ind,
                         normal_map_path=normal_map_path)
