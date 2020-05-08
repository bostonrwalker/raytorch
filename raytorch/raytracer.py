from raytorch.util import auto_str, clamp
from raytorch.core import RGB, Black, Vector
from raytorch.surface import Surface, Parallelogram, Sphere
from raytorch.texture import Texture, ImageTexture, FresnelTexture
from raytorch.camera import Camera
from raytorch.renderer import Renderer
from typing import Iterable
from PIL import Image
from typing import List, Sequence
from abc import ABC, abstractmethod
import math
from itertools import chain
from functools import reduce


class RayTracer(Renderer):

    # A big number
    RAY_LEN = 1e1
    # A small number
    RAY_OFFSET_LEN = 1e-3

    # Default colour for fully-attenuated or non-intersecting rays
    RAY_DEFAULT_COL = Black

    # Max recursion depth
    RAY_MAX_DEPTH = 16
    # Minimum weight
    RAY_MIN_WEIGHT = 1 / 256

    class ColourFunc(ABC):

        @abstractmethod
        def eval(self) -> RGB:
            pass

    class ConstCol(ColourFunc):
        """
        Represent a constant colour (i.e. not dependent on secondary effects)
        """

        def __init__(self, col: RGB):
            self.col = col

        def eval(self):
            return self.col

    class Sum(ColourFunc):
        """
        A summation of colour functions
        """

        def __init__(self, fns):
            """
            :param fns: Colour functions
            """
            self.fns = fns

        def eval(self):
            return reduce(lambda a, b: a + b, [f.eval() for f in self.fns], RayTracer.RAY_DEFAULT_COL)

    @auto_str
    class Ray(ColourFunc):

        @auto_str
        class Interaction(object):

            def __init__(self, t: int, u: float, v: float, c: Vector, r: Vector, n: Vector):
                """
                :param t: Index of texture interacted with
                :param u, v: Interaction point in texture coordinates
                :param c: Interaction point in eye space
                :param r: Unit incident vector
                :param n: Unit normal vector
                """
                self.t = t
                self.u = u
                self.v = v
                self.c = c
                self.r = r
                self.n = n

        def __init__(self, src: Vector, dst: Vector, weight: float = 1., depth: int = 0, col_fn=None, children=None,
                     i: Interaction = None):
            """
            :param src: Source point in eye space
            :param dst: Destination point in eye space
            :param weight: Ray colour weighting
            :param depth: Recursion depth
            :param col_fn: Colour function
            :param i: Interaction at ray endpoint
            """
            if col_fn is None:
                col_fn = RayTracer.ConstCol(RayTracer.RAY_DEFAULT_COL)
            if children is None:
                children = []
            self.src = src
            self.dst = dst
            self.weight = weight
            self.depth = depth
            self.col_fn = col_fn
            self.children = children
            self.i = i

        def eval(self):
            """
            Evaluate colour func at intersection (implementation)
            """
            # TODO: implement fog attenuation
            return self.col_fn.eval()

    def __init__(self):
        self.textures = []
        self.fov = 0.
        self.width = 0
        self.height = 0

    def load_textures(self, textures: Sequence[Texture]):
        self.textures = textures

    def set_fov(self, fov: float, width: int, height: int, **kwargs):
        self.fov = fov
        self.width = width
        self.height = height

    def render(self, camera: Camera, world: Iterable[Surface], **kwargs) -> Image:

        aspect_ratio = self.width / self.height

        # Field of view in radians
        horizontal_fov = self.fov * math.pi / 180.0
        vertical_fov = horizontal_fov / aspect_ratio

        # Expand rays in fish-eye array
        pixel_rays = [None] * self.width * self.height
        for j in range(self.height):
            phi = (j + 0.5 - self.height // 2) * (vertical_fov / self.height)
            for i in range(self.width):
                theta = (i + 0.5 - self.width // 2) * (horizontal_fov / self.width)
                origin = Vector.zeros()
                tip = Vector(math.sin(theta), math.sin(phi) * math.cos(theta),
                             -math.cos(theta) * math.cos(phi)) * RayTracer.RAY_LEN
                # noinspection PyTypeChecker
                pixel_rays[(self.height - j - 1) * self.width + i] = RayTracer.Ray(origin, tip)

        # Apply camera view matrix to world
        view = camera.look_at()
        world = [view @ surface for surface in world]

        # Ray tracing algorithm (breadth-first tree traversal)
        rays = [r for r in pixel_rays]
        while len(rays) > 0:

            # Find first interaction for each ray
            for s in world:
                for r in rays:

                    # noinspection PyTypeChecker

                    i = RayTracer.intersect(r, s)

                    if i is not None:

                        r.i = i
                        # Interaction point becomes new ray dst
                        r.dst = i.c

            # Implement interactions
            for r in rays:
                if r.i is not None and r.depth < RayTracer.RAY_MAX_DEPTH:
                    r.children, r.col_fn = RayTracer.interact(self.textures[r.i.t], r.i, r.weight, r.depth)

            # Add new rays created back to buffer
            rays = list(chain.from_iterable([r.children for r in rays]))

        # Rasterize
        pixels = [r.eval().asinttuple() for r in pixel_rays]
        image = Image.new('RGB', (self.width, self.height))
        image.putdata(pixels)

        return image

    @staticmethod
    def intersect(r: Ray, s: Surface) -> (float, float, float, Vector, Vector, Vector):
        """
        Find an intersection point between this ray and a surface in eye space.
        Return None if no intersection exists.

        :return: tuple containing following values:
            (float) Ray intersection parameterization in range [0, 1),
            (float) Surface intersection parameterization u in range [0, 1),
            (float) Surface intersection parameterization v in range [0, 1),
            (Vector) Ray incidence point in eye space (i.e. new ray.dst),
            (Vector) Unit incident vector in eyespace,
            (Vector) Unit normal vector of surface at incidence
        """

        if isinstance(s, Parallelogram):
            return RayTracer._intersect_parallelogram(r, s)
        elif isinstance(s, Sphere):
            return RayTracer._intersect_sphere(r, s)
        raise TypeError(f'Unsupported surface type: {type(s)}')

    @staticmethod
    def _intersect_parallelogram(r: Ray, s: Parallelogram) -> (float, float, float, Vector, Vector, Vector):
        """
        Ray equations - t in range [0, 1)
        x = r.src.x + t * (r.dst.x - r.src.x)
        y = r.src.y + t * (r.dst.y - r.src.y)
        z = r.src.z + t * (r.dst.z - r.src.z)

        Plane equation
        n.x * (x - s.pos.x) + n.y * (y - s.pos.y) + n.z * (z - s.pos.z) = 0

        Solution (ray parameterization)
        t = n @ (s.pos - r.src) / n @ (r.dst - r.src)
        """

        # Surface normal vector
        n = s.normal_vector()

        # Ray vector (tail to tip)
        d = r.dst - r.src
        dotn = n @ d

        # Check if surface facing towards ray
        if dotn < 0:

            # Find point along ray parameterization where intersection occurs
            t = (n @ (s.pos - r.src)) / dotn

            # Check if intersection between tail and tip
            if 0. <= t < 1.:

                # Find intersection point in world coordinates
                c = r.src + t * d

                # Find intersection point in parallelogram texture coordinates
                cp = (c - s.pos)
                u, v = cp @ s.bottom_edge, cp @ s.left_edge

                # Check if collision point within plane
                if 0. <= u < 1. and 0. <= v < 1.:

                    return RayTracer.Ray.Interaction(s.texture, u, v, c, d / d.norm(), n)

        # No intersection
        return None

    @staticmethod
    def _intersect_sphere(r: Ray, s: Sphere) -> (float, float, float, Vector, Vector, Vector):
        """
        Ray equations - t in range [0, 1)
        x = r.src.x + t * (r.dst.x - r.src.x)
        y = r.src.y + t * (r.dst.y - r.src.y)
        z = r.src.z + t * (r.dst.z - r.src.z)

        Sphere equation
        (x - s.pos.x) ^ 2 + (y - s.pos.y) ^ 2 + (z - s.pos.z) ^ 2 - s.radius = 0

        Solution (ray parameterization)
        d = r.dst - r.src
        u = ((s.pos - r.src) @ d) / d.norm()
        t = (u +/- math.sqrt(u ** 2 + s.radius - self.src.norm() ** 2 - s.pos.norm() ** 2)) / d.norm()
        """

        # Ray vector (tail to tip)
        d = r.dst - r.src
        magd = d.norm()

        # Sphere direction vector
        p = s.pos - r.src

        # Semi-normalized dot product between sphere direction and ray direction
        u = (p @ d) / magd

        # If sphere is not inverted, centre must be ahead of ray
        if u >= 0. or s.inverted:

            # Quadratic determinant
            det = u ** 2 + s.radius ** 2 - p.norm() ** 2

            # Check if quadratic equation has real solutions
            if det >= 0.:

                # Get negative solution if sphere not inverted, positive solution if sphere inverted
                t = (u - math.sqrt(det)) / magd if not s.inverted else (u + math.sqrt(det)) / magd

                # If intersection occurs between tip and tail
                if 0. <= t < 1.:

                    # Find intersection point in world coordinates
                    c = r.src + t * d

                    # Get normal vector at intersection
                    n = c - s.pos if not s.inverted else s.pos - c
                    n /= n.norm()

                    # TODO: implement texture coords
                    u, v = .0, .0
                    return RayTracer.Ray.Interaction(s.texture, u, v, c, d / magd, n)

        return None

    @staticmethod
    def interact(t: Texture, i: Ray.Interaction, weight: float, depth: int) -> (List[Ray], ColourFunc):
        """
        Interact ray with surface, returning child rays and a colour function
        :param t: (Texture)
        :param i: (RayTracer.Ray.Interaction)
        :param weight: Ray weight
        :param depth: Ray depth
        """
        if isinstance(t, ImageTexture):
            return RayTracer._interact_image(t, i, weight)
        elif isinstance(t, FresnelTexture):
            return RayTracer._interact_fresnel(t, i, weight, depth)
        else:
            raise TypeError(f'Unsupported texture type: {type(i.t)}')

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _interact_image(t: Texture, i: Ray.Interaction, weight: float):
        texel = t.texel_at(i.u, i.v)
        return [], RayTracer.ConstCol(texel * weight)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _interact_fresnel(t: Texture, i: Ray.Interaction, weight: float, depth: int):

        # Reflection
        rdotn = (i.r @ i.n)
        ref = i.r - 2 * rdotn * i.n
        theta_r = math.acos(clamp(ref @ i.n, 0., 1.))

        # Transmission/refraction
        ncrossr = i.n.cross(i.r)
        trans = i.n.cross(-ncrossr) / t.refraction_ind - \
                math.sqrt(1 - (ncrossr @ ncrossr) / t.refraction_ind ** 2) * i.n

        # Reflectivity function
        pr, pt, pa = t.reflectivity_func(theta_r)

        # Attenuate ray
        pr *= weight
        pt *= weight
        pa *= weight

        rays = []
        fns = []

        # Reflected ray
        if pr > RayTracer.RAY_MIN_WEIGHT:
            ref_ray = RayTracer.Ray(i.c + i.n * RayTracer.RAY_OFFSET_LEN, i.c + ref * RayTracer.RAY_LEN,
                                    weight=pr, depth=depth+1, col_fn=RayTracer.ConstCol(t.surface_col))
            rays.append(ref_ray)
            fns.append(ref_ray)

        # Transmitted ray
        if pt > RayTracer.RAY_MIN_WEIGHT:
            trans_ray = RayTracer.Ray(i.c - i.n * RayTracer.RAY_OFFSET_LEN, i.c + trans * RayTracer.RAY_LEN,
                                      weight=pt, depth=depth+1, col_fn=RayTracer.ConstCol(t.surface_col))
            rays.append(trans_ray)
            fns.append(trans_ray)

        # Ambient colouration
        if pa > RayTracer.RAY_MIN_WEIGHT:
            fns.append(RayTracer.ConstCol(t.surface_col * pa))

        return rays, RayTracer.Sum(fns)
