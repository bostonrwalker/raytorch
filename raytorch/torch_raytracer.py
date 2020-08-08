from raytorch.core import Black
from raytorch.surface import Surface, Parallelogram, Ellipsoid
from raytorch.texture import Texture, ImageTexture, FresnelTexture
from raytorch.camera import Camera
from raytorch.renderer import Renderer
from typing import Iterable
from PIL import Image
from typing import Sequence
import math
import torch


class Tensor(torch.Tensor):
    """
    Custom initialization of torch.Tensor
    """
    def __new__(cls, *args, dtype=torch.float32, device=None, **kwargs):
        if 'size' in kwargs.keys():
            # Pre-allocated tensor initialization
            if 'fill_value' in kwargs.keys():
                obj = torch.full(*args, **kwargs, requires_grad=False, dtype=dtype)
            else:
                obj = torch.Tensor(*args, **kwargs, requires_grad=False, dtype=dtype)
        else:
            obj = torch.tensor(*args, **kwargs, requires_grad=False, dtype=dtype)
        obj = obj.to(device)
        obj.__class__ = Tensor
        return obj

    # noinspection PyMissingConstructor
    def __init__(self, *args, **kwargs):
        self._init_shape = list(self.shape[1:])

    def clone(self, *args, **kwargs):
        obj = super().clone(*args, **kwargs)
        obj.__class__ = self.__class__
        obj._init_shape = self._init_shape
        return obj

    def __getitem__(self, key):
        obj = super().__getitem__(key)
        if isinstance(obj, torch.Tensor):
            # Keep Tensors as Tensors when possible
            if self.__class__ == Tensor:
                obj._init_shape = list(self.shape[1:])
                obj.__class__ = Tensor
            # Keep Tensor subclasses as Tensor subclasses when shape remains the same
            elif issubclass(self.__class__, Tensor) and len(obj.shape[1:]) == len(self._init_shape) and \
                    list(obj.shape[1:]) == self._init_shape:
                obj._init_shape = list(self.shape[1:])
                obj.__class__ = self.__class__
        return obj

    @staticmethod
    def cat(tensors, cls: type, *args, **kwargs):
        if not issubclass(cls, Tensor):
            raise ValueError('cls must be a subclass of Tensor')
        try:
            for t in tensors:
                if not isinstance(t, cls):
                    raise ValueError(f'All tensors must be instances of {cls}')
        except TypeError:
            raise ValueError('tensors must be an iterable')
        obj = torch.cat(tensors, *args, **kwargs)
        # noinspection PyProtectedMember
        obj._init_shape = tensors[0]._init_shape
        obj.__class__ = cls
        return obj


class RayTensor(Tensor):
    """
    Float tensor containing ray data - [N rays x 3 dimensions x 7 vectors]:
        (0) s: Source vector in eye space
        (1) d: Destination vector in eye space
        (2) t: Tangent (+x) vector at surface interaction in eye space
        (3) b: Bitangent (+y) vector at surface interaction in eye space
        (4) n: Normal vector at surface interaction in eye space
        (5) uv:
            (0-1) 2-D coords of interaction point in texture space
            (2): Ray weight
        (6) col: RGB colour vector

    Note: n = t x b, (t, b, n) form basis for surface tangent space
    """
    __DIM = [7, 3]

    def __new__(cls, n, dtype=torch.float32, **kwargs):
        if dtype not in [torch.float16, torch.float32, torch.float64, torch.float]:
            raise ValueError(f'Invalid dtype for RayTensor: {dtype}')
        obj = Tensor(size=[n] + RayTensor.__DIM, fill_value=0., dtype=dtype, **kwargs)
        obj.__class__ = RayTensor
        return obj

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def cat(tensors, *args, **kwargs):
        return Tensor.cat(tensors, cls=RayTensor, *args, **kwargs)

    @property
    def src(self):
        return self[:, 0, :]

    @src.setter
    def src(self, x):
        self[:, 0, :] = x

    @property
    def dst(self):
        return self[:, 1, :]

    @dst.setter
    def dst(self, x):
        self[:, 1, :] = x

    @property
    def tan(self):
        return self[:, 2, :]

    @tan.setter
    def tan(self, x):
        self[:, 2, :] = x

    @property
    def bit(self):
        return self[:, 3, :]

    @bit.setter
    def bit(self, x):
        self[:, 3, :] = x

    @property
    def nrm(self):
        return self[:, 4, :]

    @nrm.setter
    def nrm(self, x):
        self[:, 4, :] = x

    @property
    def tbn(self):
        return self[:, 2:5, :]

    @tbn.setter
    def tbn(self, x):
        self[:, 2:5, :] = x

    @property
    def uv(self):
        return self[:, 5, 0:2]

    @uv.setter
    def uv(self, x):
        self[:, 5, 0:2] = x

    @property
    def wgt(self):
        return self[:, 5, 2]

    @wgt.setter
    def wgt(self, x):
        self[:, 5, 2] = x

    @property
    def col(self):
        return self[:, 6, :]

    @col.setter
    def col(self, x):
        self[:, 6, :] = x


class LinkTensor(Tensor):
    """
    Long tensor containing linkage data (parent ray, interaction texture) - [N rays x 2 entries]:
        (0) Parent ray index (-1 = NULL)
        (1) Interaction texture index (-1 = NULL)
    """
    __DIM = [2]

    def __new__(cls, n, dtype=torch.int64, **kwargs):
        if dtype not in [torch.int32, torch.int64, torch.int]:
            # Shouldn't use less than 32-bit integers to track linkages
            raise ValueError(f'Unsupported/invalid dtype for LinkTensor: {dtype}')
        obj = Tensor(size=[n] + LinkTensor.__DIM, fill_value=-1, dtype=dtype, **kwargs)
        obj.__class__ = LinkTensor
        return obj

    @staticmethod
    def cat(tensors, *args, **kwargs):
        return Tensor.cat(tensors, cls=LinkTensor, *args, **kwargs)

    @property
    def par(self):
        return self[:, 0]

    @par.setter
    def par(self, x):
        self[:, 0] = x

    @property
    def tex(self):
        return self[:, 1]

    @tex.setter
    def tex(self, x):
        self[:, 1] = x


# Custom batch_dot() method for tensors
def _batch_dot(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.bmm(self.unsqueeze(1), other.unsqueeze(2)).squeeze(2).squeeze(1)
torch.Tensor.batch_dot = _batch_dot


# noinspection PyMethodMayBeStatic
class RayTracer(Renderer):

    def __init__(self, max_depth=16, ray_min_weight=1/256., ray_col=Black, ray_len=1e1, ray_offset_len=5e-3,
                 fresnel_func_res=128, device=None, float_dtype=torch.float32, int_dtype=torch.int64):
        """
        :param max_depth: Maximum ray tracing recursion depth
        :param ray_min_weight: Minimum ray colour weighting
        :param ray_col: Default ray colour (background colour)
        :param ray_len: Ray initialization length (must be larger than distances within scene, but as small as possible)
        :param ray_offset_len: Reflection/refraction offsetting length (must be smaller than distances within scene, but
                               as large as possible)
        :param fresnel_func_res: Interpolation resolution for Fresnel reflectivity functions
        :param device: PyTorch device (default: CUDA:0 if CUDA available, else CPU)
        """

        self.max_depth = max_depth
        self.ray_min_weight = ray_min_weight
        self.ray_col = ray_col
        self.ray_len = ray_len
        self.ray_offset_len = ray_offset_len
        self.fresnel_func_res = fresnel_func_res

        # Default: check if GPU is available
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype

        self.textures = []
        self.ambient_maps = {}
        self.normal_maps = {}
        self.fresnel_funcs = {}
        self.width, self.height = 0, 0
        self.fov = (self._ray_tensor(0), self._link_tensor(0))

    def load_textures(self, textures: Sequence[Texture]):
        self.textures = textures
        for i in range(len(textures)):
            texture = textures[i]
            if isinstance(texture, ImageTexture):
                self._load_image_texture(texture)
            elif isinstance(texture, FresnelTexture):
                self._load_fresnel_texture(texture)

    def _load_image_texture(self, texture: ImageTexture):
        self.ambient_maps[texture] = self._image_as_tensors(texture.ambient)

    def _load_fresnel_texture(self, texture: FresnelTexture):
        self.fresnel_funcs[texture] = self._float_tensor([list(texture.reflectivity_func(
            (step / self.fresnel_func_res) * math.pi / 2)) for step in range(self.fresnel_func_res)])
        if texture.normal_map is not None:
            self.normal_maps[texture] = self._normal_map_as_tensors(texture.normal_map)

    def _image_as_tensors(self, image: Image):
        width, height = image.width, image.height
        pixels = list([p[0:3] for p in image.getdata()])
        rgb = self._float_tensor(pixels) / 255.
        return self._float_tensor([width, height]), rgb.reshape([height, width, 3]).flip(0).transpose(0, 1)

    def _normal_map_as_tensors(self, image: Image):
        width, height = image.width, image.height
        pixels = list([p[0:3] for p in image.getdata()])
        xyz = self._float_tensor(pixels) / 128. - 1.
        xyz = xyz / xyz.norm(dim=1).unsqueeze(1)
        return self._float_tensor([width, height]), xyz.reshape([height, width, 3]).flip(0).transpose(0, 1)

    def _get_texels(self, texture: ImageTexture, uv: torch.Tensor):
        """
        Return pixel colours as Tensor
        :param uv: [N x 2] Texture coordinates
        """
        dim, pixels = self.ambient_maps[texture]
        p = (uv * dim).floor().long()
        return pixels[p[:, 0], p[:, 1]]

    def _get_normals(self, texture: FresnelTexture, uv: torch.Tensor):
        """
        Return normals as Tensor
        :param uv: [N x 2] Texture coordinates
        """
        if texture in self.normal_maps.keys():
            dim, normals = self.normal_maps[texture]
            p = (uv * dim).floor().long()
            return normals[p[:, 0], p[:, 1]]
        else:
            return None

    def _get_reflectivity(self, texture: FresnelTexture, theta: torch.Tensor):
        """
        Get reflectivity
        :param texture: (FresnelTexture)
        :param theta: (torch.Tensor [N]) Reflection angle
        :return: (torch.Tensor [N x 3]) reflection/refraction/ambient proportions
        """
        indices = ((theta / (math.pi / 2)) * self.fresnel_func_res).floor().long().clamp(0, self.fresnel_func_res - 1)
        return self.fresnel_funcs[texture][indices, :]

    def set_fov(self, fov: float, width: int, height: int, **kwargs):

        self.width, self.height = width, height

        # Build and cache starting point for FOV rays in eye space
        self.fov = self._build_spherical_perspective(fov, width, height)

    def render(self, camera: Camera, world: Iterable[Surface], **kwargs) -> Image:

        # Start with cached FOV
        pixel_rays, pixel_ray_links = self.fov[0].clone(), self.fov[1].clone()

        # Apply camera view matrix to world
        view = camera.look_at()
        world = [view @ surface for surface in world]

        # Ray tracing algorithm (breadth-first tree traversal)
        tree = {0: (pixel_rays, pixel_ray_links)}

        for gen in range(0, self.max_depth):

            # Simulate intersections with each surface
            rays, links = tree[gen]
            for surface in world:
                self.intersect(rays, links, surface)

            # Simulate interactions with each texture
            new_ray_tensors, new_link_tensors = [], []
            for texture_ind in links.tex.unique():
                if texture_ind >= 0:
                    texture = self.textures[texture_ind]
                    # noinspection PyUnresolvedReferences
                    ray_ind = (links.tex == texture_ind).nonzero().squeeze(1)
                    new_rays, new_links = self.interact(rays, ray_ind, texture)
                    if new_rays.shape[0] > 0:
                        new_ray_tensors.append(new_rays)
                        new_link_tensors.append(new_links)

            # Continue only if any rays generated
            if len(new_ray_tensors) > 0:
                tree[gen + 1] = (RayTensor.cat([self._ray_tensor(0)] + new_ray_tensors),
                                 LinkTensor.cat([self._link_tensor(0)] + new_link_tensors))
                continue
            else:
                break

        # Rasterize
        for gen in reversed(list(tree.keys())[1:]):

            # Add colour back to parent rays
            rays, links = tree[gen]
            par_rays = tree[gen - 1][0]
            # noinspection PyUnresolvedReferences
            par_rays.col.index_add_(0, links.par, rays.col)

        # Convert
        pixels = (pixel_rays.col * 256.).floor().int().clamp(0, 255)

        image = Image.new('RGB', (self.width, self.height))
        image.putdata([tuple(p) for p in pixels.tolist()])

        return image

    def intersect(self, rays: RayTensor, links: LinkTensor, surface: Surface):
        """

        """
        if isinstance(surface, Parallelogram):
            self._intersect_parallelogram(rays, links, surface)
        elif isinstance(surface, Ellipsoid):
            self._intersect_ellipsoid(rays, links, surface)
        else:
            raise TypeError(f'Unsupported surface type: {type(surface)}')

    def _intersect_parallelogram(self, rays: RayTensor, links: LinkTensor, surface: Parallelogram):
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

        # Surface description
        pos = self._float_tensor(surface.pos.aslist()[0:3])
        btm = self._float_tensor(surface.bottom_edge.aslist()[0:3])
        left = self._float_tensor(surface.left_edge.aslist()[0:3])

        # Unit normal vector
        n = btm.cross(left)
        n /= n.norm()

        # Ray direction vectors
        d = (rays.dst - rays.src).squeeze(dim=1)

        # Ray lengths
        l = d.norm(dim=1)

        # Ray unit direction vectors
        r = d / l.unsqueeze(-1)

        # Cosine of ray with surface
        dotn = r @ n

        # Check if surface facing towards ray
        ind0 = (dotn < 0.).nonzero().squeeze(1)

        # Find point along ray parameterization where intersection occurs
        t = ((pos - rays.src[ind0].squeeze(1)) @ n) / (l[ind0] * dotn[ind0])

        # Check if intersection between tail and tip
        ind1 = ((t >= 0.) & (t < 1.)).nonzero().squeeze(-1)
        ind0 = ind0[ind1]

        # Find intersection point in world coordinates
        c = rays.src[ind0] + t[ind1].unsqueeze(-1) * d[ind0]

        # Find intersection point in parallelogram texture coordinates
        cp = c - pos
        uv = cp @ torch.cat((btm.unsqueeze(-1), left.unsqueeze(-1)), dim=1)

        # Check if collision point within plane
        ind2 = ((uv[:, 0] >= 0.) & (uv[:, 0] < 1.) & (uv[:, 1] >= 0.) & (uv[:, 1] < 1.)).nonzero().squeeze(1)
        ind0 = ind0[ind2]

        # Tangent and bitangent
        tan = btm / btm.norm(dim=0)
        bit = left / left.norm(dim=0)

        # Modify values
        rays.dst[ind0] = c[ind2, :]  # New destination points
        rays.tan[ind0] = tan  # Tangent
        rays.bit[ind0] = bit  # Bitangent
        rays.nrm[ind0] = n  # Normal
        rays.uv[ind0] = uv[ind2, :]  # Texture coords

        links.tex[ind0] = surface.texture  # Texture index

    # noinspection PyPep8Naming
    def _intersect_ellipsoid(self, rays: RayTensor, links: LinkTensor, ellipsoid: Ellipsoid):

        # Ellipsoid centrepoint
        pos = self._float_tensor(ellipsoid.pos.aslist()[0:3])

        """
        Matrix composition
        Q = A'A
        A = RS
        R = [ax1/|ax1| ax2/|ax2| ax3/|ax3|]': model rotation matrix (orthogonal)
        R^-1 = R' = [ax1/|ax1| ax2/|ax2| ax3/|ax3|]
        S = diag(1/|ax1| 1/|ax2| 1/|ax3|): model scaling matrix (diagonal)
        S^-1 = 1/S = diag(|ax1| |ax2| |ax3|)
        A = [ax1/|ax1|^2 ax2/|ax2|^2 ax3/|ax3|^2]: eye space -> model space
        A^-1 = S^-1 R^-1 = [ax1 ax2 ax3]': model space -> eye space
        """

        # Axes as transformation matrix (rotation/scaling) [3 x 3]
        A = self._float_tensor([ellipsoid.first_axis.aslist()[0:3], ellipsoid.second_axis.aslist()[0:3],
                                ellipsoid.third_axis.aslist()[0:3]])
        A = A / (A.norm(dim=1) ** 2).unsqueeze(-1)
        A_t = A.transpose(0, 1)
        A_inv_t = self._float_tensor([ellipsoid.first_axis.aslist()[0:3], ellipsoid.second_axis.aslist()[0:3],
                                      ellipsoid.third_axis.aslist()[0:3]])
        Q = A_t @ A

        """
        Ray equations - t in range [0, 1)
        x = src + t * dir

        Ellipsoid equation
        [A(x - pos)]'[A(x - pos)] = (x - pos)'A'A(x - pos) = (x - pos)' Q (x - pos) = 1, Q = A' A

        Solution (ray parameterization)
        p = src - pos  # Shorthand
        (p + t dir) A (p + t dir)' = 1  # Substituting
        (dir A dir') t^2 + 2 (p A dir') t + p A p' - 1 = 0  # Quadratic form
        u = -(p A dir')  # Midpoint
        det = [u^2 - (dir A dir') (p A p' - 1)]  # Quadratic discriminant / 4
        t = [u +/- sqrt(det)] / (dir A dir')
        """

        # Ray vectors
        d = (rays.dst - rays.src)

        # Normalized projection of ellipsoid centrepoint onto ray
        p = rays.src - pos
        Q_dot_d = d @ Q
        u = -_batch_dot(p, Q_dot_d)

        # If ellipsoid is not inverted, centre must be ahead of ray
        ind0 = (u >= 0.).nonzero().squeeze(1) if not ellipsoid.inverted else torch.arange(rays.shape[0])
        d = d[ind0]
        Q_dot_d = Q_dot_d[ind0]
        p = p[ind0]
        u = u[ind0]

        # Projection magnitude of ellipsoid centrepoint onto ray
        mag_d = _batch_dot(d, Q_dot_d)
        mag_p = _batch_dot(p, p @ Q)

        # Quadratic discriminant
        det = u ** 2 - (mag_p - 1.) * mag_d

        # Check if quadratic equation has real solutions
        ind1 = (det >= 0.).nonzero().squeeze(1)
        ind0 = ind0[ind1]
        d = d[ind1]
        u = u[ind1]
        det = det[ind1]
        mag_d = mag_d[ind1]

        # Get negative solution if sphere not inverted, positive solution if sphere inverted
        t = (u - det.sqrt()) / mag_d if not ellipsoid.inverted else (u + det.sqrt()) / mag_d

        # If intersection occurs between tip and tail
        ind2 = ((t >= 0.) & (t < 1.)).nonzero().squeeze(1)
        ind0 = ind0[ind2]
        d = d[ind2]
        t = t[ind2]

        # Find intersection point in world coordinates
        c = rays.src[ind0] + t.unsqueeze(-1) * d

        """
        TBN space at intersection
        
        n = [cos(phi)sin(theta), sin(phi)sin(theta), cos(theta)]
        |n|^2 = cos^2(phi)sin^2(theta) + sin^2(phi)sin^2(theta) + cos^2(theta) = sin^2(theta) + cos^2(theta)
              = 1

        cos(theta) = nz
        sin(theta) = sqrt(1 - nz^2)
        cos(phi) = nx / sin(theta)
        sin(phi) = ny / sin(theta)

        t = [-sin(phi), cos(phi), 0]
          = [-ny / sin(theta), nx / sin(theta), 0]
        |t|^2 = sin^2(phi) + cos^2(phi) = 1

        b = n x t = [ny*tz - ty*nz, nz*tx - tz*nx, nx*ty - tx*ny]
                  = [sin(phi)sin(theta)*0 - cos(phi)*cos(theta), cos(theta)*-sin(phi) - 0*cos(phi)sin(theta),
                     cos(phi)sin(theta)*cos(phi) - -sin(phi)*sin(phi)sin(theta)]
                  = [-cos(phi)cos(theta), -sin(phi)cos(theta), (cos^2(phi)+sin^2(phi))sin(theta)]
                  = [-cos(phi)cos(theta), -sin(phi)cos(theta), sin(theta)]

        b = [-cos(phi)cos(theta), -sin(phi)cos(theta), sin(theta)]
          = [-nx * nz / sin(theta), -ny * nz / sin(theta), sin(theta)]
          
        """

        # Normal
        n = (c - pos) @ A_t if not ellipsoid.inverted else (pos - c) @ A_t
        n /= n.norm(dim=1).unsqueeze(-1)

        # Intermediate terms derived from normal
        # noinspection PyTypeChecker,PyUnresolvedReferences
        sintheta = (1 - n[:, 2] ** 2).sqrt()
        cottheta = n[:, 2] / sintheta

        # Tangent
        # noinspection PyArgumentList
        t = torch.cat([(-n[:, 1] / sintheta).unsqueeze(1), (n[:, 0] / sintheta).unsqueeze(1),
                       self._float_tensor(size=[n.shape[0], 1], fill_value=0.)], axis=1)

        # Bitangent
        # noinspection PyArgumentList
        b = torch.cat([(-n[:, 0] * cottheta).unsqueeze(1), (-n[:, 1] * cottheta).unsqueeze(1),
                       sintheta.unsqueeze(1)], axis=1)

        # noinspection PyArgumentList
        tbn = torch.cat([t.unsqueeze(1), b.unsqueeze(1), n.unsqueeze(1)], axis=1)
        tbn_eye = tbn @ A_inv_t

        """
        Spherical coordinates at intersection
        x' = Ax
        phi = atan(x'[1] / x'[0])
        theta = acos(x'[2])
        
        UV coordinates at intersection
        uv = [phi / 2 pi, theta / pi]
        """

        c_prime = n if not ellipsoid.inverted else -n
        phi, theta = c_prime[:, 1].atan2(c_prime[:, 0]), torch.acos(c_prime[:, 2])
        # noinspection PyArgumentList
        uv = torch.cat([(phi / (2 * math.pi)).unsqueeze(1), (theta / math.pi).unsqueeze(1)], dim=1)

        """
        Modify values in-place
        """

        rays.dst[ind0] = c  # New destination points
        rays.tbn[ind0] = tbn_eye  # Tangent, bitangent, normal
        rays.uv[ind0] = uv  # Texture coordinates

        links.tex[ind0] = ellipsoid.texture  # Texture index

    def interact(self, rays: RayTensor, indices: torch.IntTensor, texture: Texture) -> (RayTensor, LinkTensor):
        """
        """
        if isinstance(texture, ImageTexture):
            return self._interact_image(rays, indices, texture)
        elif isinstance(texture, FresnelTexture):
            return self._interact_fresnel(rays, indices, texture)
        else:
            raise TypeError(f'Unsupported texture type: {type(texture)}')

    def _interact_image(self, rays: RayTensor, indices: torch.IntTensor, texture: ImageTexture) ->\
            (torch.Tensor, torch.LongTensor):
        rays.col[indices] = self._get_texels(texture, rays.uv[indices]) * rays.wgt[indices].unsqueeze(-1)
        return self._ray_tensor(0), self._link_tensor(0)

    def _interact_fresnel(self, rays: RayTensor, indices: torch.IntTensor, texture: FresnelTexture) ->\
            (torch.Tensor, torch.LongTensor):

        # Subset parent rays only
        par_rays = rays[indices]

        # Ray direction vectors
        d = (par_rays.dst - par_rays.src).squeeze(dim=1)
        d /= d.norm(dim=1).unsqueeze(-1)

        # Model normal
        model_n = par_rays.nrm

        # Surface normal vectors
        if texture in self.normal_maps.keys():
            # Use mapped normal vectors
            n_prime = self._get_normals(texture, par_rays.uv)
            n = (n_prime.unsqueeze(1) @ par_rays.tbn).squeeze(1)
        else:
            # Use surface normal
            n = model_n

        # Normalize normal -_-
        n /= n.norm(dim=1).unsqueeze(-1)

        # Intermediate products
        ddotn = _batch_dot(d, n)

        # Incident angle
        theta = torch.acos(_batch_dot(-d, n).clamp(0., 1.))

        # Reflection vector
        ref = d - 2. * ddotn.unsqueeze(-1) * n

        # Transmission vector
        eta = texture.refraction_ind
        trans = (torch.cross(torch.cross(n, d), n) - (eta ** 2 - torch.sin(theta) ** 2).sqrt().unsqueeze(-1) * n) / eta

        # Reflectivity function
        reflectivity = self._get_reflectivity(texture, theta) * par_rays.wgt.unsqueeze(-1)
        pr, pt, pa = reflectivity[:, 0], reflectivity[:, 1], reflectivity[:, 2]

        """
        Ambient colouration
        """
        # Ambient surface colour
        col = self._float_tensor(texture.surface_col.asarray())

        # Apply ambient colouration
        rays.col[indices] = pa.unsqueeze(-1) * col.unsqueeze(0).expand(indices.shape[0], 3)
        
        """
        Reflection
        """
        # Enforce minimum weight of reflected ray
        ref_ind = (pr > self.ray_min_weight).nonzero().squeeze(1)  # Reflected ray indices in subset
        ref_par_rays = par_rays[ref_ind, :, :]
        ref_wgt = pr[ref_ind]
        ref_col = ref_wgt.unsqueeze(-1) * col.unsqueeze(0).expand(ref_wgt.shape[0], 3)

        # Construct ray, links datasets
        ref_rays, ref_links = self._ray_tensor(ref_ind.shape[0]), self._link_tensor(ref_ind.shape[0])
        ref_rays.src = ref_par_rays.dst + model_n[ref_ind, :] * self.ray_offset_len
        ref_rays.dst = ref_par_rays.dst + ref[ref_ind, :] * self.ray_len
        ref_rays.wgt = ref_wgt
        ref_rays.col = ref_col
        ref_links.par = indices[ref_ind]

        """
        Transmission (refraction)
        """
        # Enforce minimum weight of transmitted ray
        trans_ind = (pt > self.ray_min_weight).nonzero().squeeze(1)  # Reflected ray indices in subset
        trans_par_rays = par_rays[trans_ind, :, :]
        trans_wgt = pt[trans_ind]
        trans_col = trans_wgt.unsqueeze(-1) * col.unsqueeze(0).expand(trans_wgt.shape[0], 3)

        # Construct ray, links datasets
        trans_rays, trans_links = self._ray_tensor(trans_ind.shape[0]), self._link_tensor(trans_ind.shape[0])
        trans_rays.src = trans_par_rays.dst - model_n[trans_ind, :] * self.ray_offset_len
        trans_rays.dst = trans_par_rays.dst + trans[trans_ind, :] * self.ray_len
        trans_rays.wgt = trans_wgt
        trans_rays.col = trans_col
        trans_links.par = indices[trans_ind]

        new_rays = RayTensor.cat([ref_rays, trans_rays])
        new_links = LinkTensor.cat([ref_links, trans_links])
        return new_rays, new_links

    def _build_spherical_perspective(self, fov: float, width: int, height: int) -> (RayTensor, LinkTensor):
        """
        Build fish-eye array of rays
        """

        aspect_ratio = width / height

        # Field of view in radians
        horizontal_fov = fov * math.pi / 180.0
        vertical_fov = horizontal_fov / aspect_ratio

        # Spherical coordinates
        phi = torch.linspace(vertical_fov / 2, -vertical_fov / 2, height, dtype=self.float_dtype)  # Declination
        theta = torch.linspace(-horizontal_fov / 2, horizontal_fov / 2, width, dtype=self.float_dtype)  # Yaw
        phi_theta = torch.cartesian_prod(phi, theta)

        # Source: origin
        src = self._float_tensor([.0, .0, .0])

        # Destination
        dst = torch.cat([
            (self.ray_len * torch.sin(phi_theta[:, 1]) * torch.cos(phi_theta[:, 0])).unsqueeze(-1),
            (self.ray_len * torch.sin(phi_theta[:, 0])).unsqueeze(-1),
            (-self.ray_len * torch.cos(phi_theta[:, 1]) * torch.cos(phi_theta[:, 0])).unsqueeze(-1)
        ], dim=1)

        rays = self._ray_tensor(width * height)
        links = self._link_tensor(width * height)
        weight = 1.

        rays.src = src
        rays.dst = dst
        rays.wgt = weight
        rays.col = self._float_tensor(self.ray_col.asarray())

        return rays, links

    def _float_tensor(self, *args, **kwargs) -> Tensor:
        return Tensor(*args, **kwargs, dtype=self.float_dtype, device=self.device)

    def _ray_tensor(self, n: int) -> RayTensor:
        # noinspection PyTypeChecker
        return RayTensor(n=n, dtype=self.float_dtype, device=self.device)

    def _link_tensor(self, n: int) -> LinkTensor:
        # noinspection PyTypeChecker
        return LinkTensor(n=n, dtype=self.int_dtype, device=self.device)
