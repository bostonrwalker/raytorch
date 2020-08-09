from raytorch.core import Black
from raytorch.surface import Surface, Parallelogram, Ellipsoid
from raytorch.texture import Texture, ImageTexture, FresnelTexture
from raytorch.camera import Camera
from raytorch.renderer import Renderer
from PIL import Image
from typing import Sequence, Iterable
import math
import torch


# Custom batch_dot() method for tensors
def _batch_dot(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.bmm(self.unsqueeze(1), other.unsqueeze(2)).squeeze(2).squeeze(1)
torch.Tensor.batch_dot = _batch_dot


class Rays:
    """
    Collection of tensors containing ray data:
        src: Source vector in eye space
        dst: Destination vector in eye space
        tbn: Tangent (+x), bitangent (+y), and normal vectors at surface interaction in eye space
            [note: n = t x b, (t, b, n) form basis for surface tangent space]
        uv: 2-D coords of interaction point in texture space
        wgt: Ray weight
        par: Link to parent rays (-1 = NULL)
        tex: Link to interaction texture (-1 = NULL)
    """
    
    def __init__(self, src: torch.Tensor, dst: torch.Tensor, tbn: torch.Tensor, uv: torch.Tensor, wgt: torch.Tensor,
                 col: torch.Tensor, par: torch.Tensor, tex: torch.Tensor):
        self.__src = src
        self.__dst = dst
        self.__tbn = tbn
        self.__uv = uv
        self.__wgt = wgt
        self.__col = col
        self.__par = par
        self.__tex = tex

    @staticmethod
    def allocate(n, float_dtype=torch.float32, int_dtype=torch.int64, device=None, **kwargs):
        if float_dtype not in [torch.float16, torch.float32, torch.float64, torch.float]:
            raise ValueError(f'Invalid float_dtype: {float_dtype}')
        if int_dtype not in [torch.int32, torch.int64, torch.int]:
            # Shouldn't use less than 32-bit integers to track linkages
            raise ValueError(f'Unsupported/invalid int_dtype: {int_dtype}')
        
        return Rays(
            src=torch.zeros(size=[n, 3], dtype=float_dtype, device=device, **kwargs),
            dst=torch.zeros(size=[n, 3], dtype=float_dtype, device=device, **kwargs),
            tbn=torch.zeros(size=[n, 3, 3], dtype=float_dtype, device=device, **kwargs),
            uv=torch.zeros(size=[n, 2], dtype=float_dtype, device=device, **kwargs),
            wgt=torch.zeros(size=[n], dtype=float_dtype, device=device, **kwargs),
            col=torch.zeros(size=[n, 3], dtype=float_dtype, device=device, **kwargs),
            par=torch.zeros(size=[n], dtype=int_dtype, device=device, **kwargs),
            tex=torch.zeros(size=[n], dtype=int_dtype, device=device, **kwargs)
        )

    @staticmethod
    def cat(tensors: Iterable, *args, **kwargs):
        return Rays(
            src=torch.cat([t.src for t in tensors], *args, **kwargs),
            dst=torch.cat([t.dst for t in tensors], *args, **kwargs),
            tbn=torch.cat([t.tbn for t in tensors], *args, **kwargs),
            uv=torch.cat([t.uv for t in tensors], *args, **kwargs),
            wgt=torch.cat([t.wgt for t in tensors], *args, **kwargs),
            col=torch.cat([t.col for t in tensors], *args, **kwargs),
            par=torch.cat([t.par for t in tensors], *args, **kwargs),
            tex=torch.cat([t.tex for t in tensors], *args, **kwargs)
        )
    
    def clone(self):
        return Rays(
            src=self.src.clone(),
            dst=self.dst.clone(),
            tbn=self.tbn.clone(),
            uv=self.uv.clone(),
            wgt=self.wgt.clone(),
            col=self.col.clone(),
            par=self.par.clone(),
            tex=self.tex.clone()
        )
    
    def __getitem__(self, item):
        return Rays(
            src=self.src[item],
            dst=self.dst[item],
            tbn=self.tbn[item],
            uv=self.uv[item],
            wgt=self.wgt[item],
            col=self.col[item],
            par=self.par[item],
            tex=self.tex[item]
        )

    def __len__(self):
        return self.__src.shape[0]

    @property
    def src(self):
        return self.__src

    @src.setter
    def src(self, x):
        self.__src[:, :] = x

    @property
    def dst(self):
        return self.__dst

    @dst.setter
    def dst(self, x):
        self.__dst[:, :] = x

    @property
    def tan(self):
        return self.__tbn[:, 0]

    @tan.setter
    def tan(self, x):
        self.__tbn[:, 0] = x

    @property
    def bit(self):
        return self.__tbn[:, 1]

    @bit.setter
    def bit(self, x):
        self.__tbn[:, 1] = x

    @property
    def nrm(self):
        return self.__tbn[:, 2]

    @nrm.setter
    def nrm(self, x):
        self.__tbn[:, 2] = x

    @property
    def tbn(self):
        return self.__tbn

    @tbn.setter
    def tbn(self, x):
        self.__tbn[:, :, :] = x

    @property
    def uv(self):
        return self.__uv

    @uv.setter
    def uv(self, x):
        self.__uv[:, :] = x

    @property
    def wgt(self):
        return self.__wgt

    @wgt.setter
    def wgt(self, x):
        self.__wgt[:] = x

    @property
    def col(self):
        return self.__col

    @col.setter
    def col(self, x):
        self.__col[:, :] = x

    @property
    def par(self):
        return self. __par

    @par.setter
    def par(self, x):
        self.__par[:] = x

    @property
    def tex(self):
        return self.__tex

    @tex.setter
    def tex(self, x):
        self.__tex[:] = x


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
        self.fov = self._allocate_rays(0)

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
        self.fresnel_funcs[texture] = self._as_float_tensor([list(texture.reflectivity_func(
            (step / self.fresnel_func_res) * math.pi / 2)) for step in range(self.fresnel_func_res)])
        if texture.normal_map is not None:
            self.normal_maps[texture] = self._normal_map_as_tensors(texture.normal_map)

    def _image_as_tensors(self, image: Image):
        width, height = image.width, image.height
        pixels = list([p[0:3] for p in image.getdata()])
        rgb = self._as_float_tensor(pixels) / 255.
        return self._as_float_tensor([width, height]), rgb.reshape([height, width, 3]).flip(0).transpose(0, 1)

    def _normal_map_as_tensors(self, image: Image):
        width, height = image.width, image.height
        pixels = list([p[0:3] for p in image.getdata()])
        xyz = self._as_float_tensor(pixels) / 128. - 1.
        xyz = xyz / xyz.norm(dim=1).unsqueeze(1)
        return self._as_float_tensor([width, height]), xyz.reshape([height, width, 3]).flip(0).transpose(0, 1)

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
        pixel_rays = self.fov.clone()

        # Apply camera view matrix to world
        view = camera.look_at()
        world = [view @ surface for surface in world]

        # Ray tracing algorithm (breadth-first tree traversal)
        tree = {0: pixel_rays}

        for gen in range(0, self.max_depth):

            # Simulate intersections with each surface
            rays = tree[gen]
            for surface in world:
                self.intersect(rays, surface)

            # Simulate interactions with each texture
            all_new_rays = []
            for texture_ind in rays.tex.unique():
                if texture_ind >= 0:
                    texture = self.textures[texture_ind]
                    # noinspection PyUnresolvedReferences
                    ray_ind = (rays.tex == texture_ind).nonzero(as_tuple=False).squeeze(1)
                    new_rays = self.interact(rays, ray_ind, texture)
                    if len(new_rays) > 0:
                        all_new_rays.append(new_rays)

            # Continue only if any rays generated
            if len(all_new_rays) > 0:
                # noinspection PyTypeChecker
                tree[gen + 1] = Rays.cat([self._allocate_rays(0)] + all_new_rays)
                continue
            else:
                break

        # Rasterize
        for gen in reversed(list(tree.keys())[1:]):

            # Add colour back to parent rays
            rays = tree[gen]
            par_rays = tree[gen - 1]
            par_rays.col.index_add_(0, rays.par, rays.col)

        # Convert
        pixels = (pixel_rays.col * 256.).floor().int().clamp(0, 255)

        image = Image.new('RGB', (self.width, self.height))
        image.putdata([tuple(p) for p in pixels.tolist()])

        return image

    def intersect(self, rays: Rays, surface: Surface):
        """

        """
        if isinstance(surface, Parallelogram):
            self._intersect_parallelogram(rays, surface)
        elif isinstance(surface, Ellipsoid):
            self._intersect_ellipsoid(rays, surface)
        else:
            raise TypeError(f'Unsupported surface type: {type(surface)}')

    def _intersect_parallelogram(self, rays: Rays, surface: Parallelogram):
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
        pos = self._as_float_tensor(surface.pos.aslist()[0:3])
        btm = self._as_float_tensor(surface.bottom_edge.aslist()[0:3])
        left = self._as_float_tensor(surface.left_edge.aslist()[0:3])

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
        ind0 = (dotn < 0.).nonzero(as_tuple=False).squeeze(1)

        # Find point along ray parameterization where intersection occurs
        t = ((pos - rays.src[ind0].squeeze(1)) @ n) / (l[ind0] * dotn[ind0])

        # Check if intersection between tail and tip
        ind1 = ((t >= 0.) & (t < 1.)).nonzero(as_tuple=False).squeeze(-1)
        ind0 = ind0[ind1]

        # Find intersection point in world coordinates
        c = rays.src[ind0] + t[ind1].unsqueeze(-1) * d[ind0]

        # Find intersection point in parallelogram texture coordinates
        cp = c - pos
        uv = cp @ torch.cat((btm.unsqueeze(-1), left.unsqueeze(-1)), dim=1)

        # Check if collision point within plane
        ind2 = ((uv[:, 0] >= 0.) & (uv[:, 0] < 1.) & (uv[:, 1] >= 0.) & (uv[:, 1] < 1.)).nonzero(as_tuple=False)\
            .squeeze(1)
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
        rays.tex[ind0] = surface.texture  # Texture index

    # noinspection PyPep8Naming
    def _intersect_ellipsoid(self, rays: Rays, ellipsoid: Ellipsoid):

        # Ellipsoid centrepoint
        pos = self._as_float_tensor(ellipsoid.pos.aslist()[0:3])

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
        A = self._as_float_tensor([ellipsoid.first_axis.aslist()[0:3], ellipsoid.second_axis.aslist()[0:3],
                                   ellipsoid.third_axis.aslist()[0:3]])
        A = A / (A.norm(dim=1) ** 2).unsqueeze(-1)
        A_t = A.transpose(0, 1)
        A_inv_t = self._as_float_tensor([ellipsoid.first_axis.aslist()[0:3], ellipsoid.second_axis.aslist()[0:3],
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
        u = -p.batch_dot(Q_dot_d)

        # If ellipsoid is not inverted, centre must be ahead of ray
        ind0 = (u >= 0.).nonzero(as_tuple=False).squeeze(1) if not ellipsoid.inverted else torch.arange(len(rays))
        d = d[ind0]
        Q_dot_d = Q_dot_d[ind0]
        p = p[ind0]
        u = u[ind0]

        # Projection magnitude of ellipsoid centrepoint onto ray
        mag_d = d.batch_dot(Q_dot_d)
        mag_p = p.batch_dot(p @ Q)

        # Quadratic discriminant
        det = u ** 2 - (mag_p - 1.) * mag_d

        # Check if quadratic equation has real solutions
        ind1 = (det >= 0.).nonzero(as_tuple=False).squeeze(1)
        ind0 = ind0[ind1]
        d = d[ind1]
        u = u[ind1]
        det = det[ind1]
        mag_d = mag_d[ind1]

        # Get negative solution if sphere not inverted, positive solution if sphere inverted
        t = (u - det.sqrt()) / mag_d if not ellipsoid.inverted else (u + det.sqrt()) / mag_d

        # If intersection occurs between tip and tail
        ind2 = ((t >= 0.) & (t < 1.)).nonzero(as_tuple=False).squeeze(1)
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
                       self._zeros([n.shape[0], 1])], axis=1)

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
        rays.tex[ind0] = ellipsoid.texture  # Texture index

    def interact(self, rays: Rays, indices: torch.IntTensor, texture: Texture) -> Rays:
        """
        """
        if isinstance(texture, ImageTexture):
            return self._interact_image(rays, indices, texture)
        elif isinstance(texture, FresnelTexture):
            return self._interact_fresnel(rays, indices, texture)
        else:
            raise TypeError(f'Unsupported texture type: {type(texture)}')

    def _interact_image(self, rays: Rays, indices: torch.IntTensor, texture: ImageTexture) -> Rays:
        rays.col[indices] = self._get_texels(texture, rays.uv[indices]) * rays.wgt[indices].unsqueeze(-1)
        return self._allocate_rays(0)

    def _interact_fresnel(self, rays: Rays, indices: torch.IntTensor, texture: FresnelTexture) -> Rays:

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
        ddotn = d.batch_dot(n)

        # Incident angle
        theta = torch.acos((-d).batch_dot(n).clamp(0., 1.))

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
        col = self._as_float_tensor(texture.surface_col.asarray())

        # Apply ambient colouration
        rays.col[indices] = pa.unsqueeze(-1) * col.unsqueeze(0).expand(indices.shape[0], 3)
        
        """
        Reflection
        """
        # Enforce minimum weight of reflected ray
        ref_ind = (pr > self.ray_min_weight).nonzero(as_tuple=False).squeeze(1)  # Reflected ray indices in subset
        ref_par_rays = par_rays[ref_ind]
        ref_wgt = pr[ref_ind]
        ref_col = ref_wgt.unsqueeze(-1) * col.unsqueeze(0).expand(ref_wgt.shape[0], 3)

        # Construct ray, links datasets
        ref_rays = self._allocate_rays(ref_ind.shape[0])
        ref_rays.src = ref_par_rays.dst + model_n[ref_ind, :] * self.ray_offset_len
        ref_rays.dst = ref_par_rays.dst + ref[ref_ind, :] * self.ray_len
        ref_rays.wgt = ref_wgt
        ref_rays.col = ref_col
        ref_rays.par = indices[ref_ind]

        """
        Transmission (refraction)
        """
        # Enforce minimum weight of transmitted ray
        trans_ind = (pt > self.ray_min_weight).nonzero(as_tuple=False).squeeze(1)  # Reflected ray indices in subset
        trans_par_rays = par_rays[trans_ind]
        trans_wgt = pt[trans_ind]
        trans_col = trans_wgt.unsqueeze(-1) * col.unsqueeze(0).expand(trans_wgt.shape[0], 3)

        # Construct ray, links datasets
        trans_rays = self._allocate_rays(trans_ind.shape[0])
        trans_rays.src = trans_par_rays.dst - model_n[trans_ind, :] * self.ray_offset_len
        trans_rays.dst = trans_par_rays.dst + trans[trans_ind, :] * self.ray_len
        trans_rays.wgt = trans_wgt
        trans_rays.col = trans_col
        trans_rays.par = indices[trans_ind]

        return Rays.cat([ref_rays, trans_rays])

    def _build_spherical_perspective(self, fov: float, width: int, height: int) -> Rays:
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
        src = self._as_float_tensor([.0, .0, .0])

        # Destination
        dst = torch.cat([
            (self.ray_len * torch.sin(phi_theta[:, 1]) * torch.cos(phi_theta[:, 0])).unsqueeze(-1),
            (self.ray_len * torch.sin(phi_theta[:, 0])).unsqueeze(-1),
            (-self.ray_len * torch.cos(phi_theta[:, 1]) * torch.cos(phi_theta[:, 0])).unsqueeze(-1)
        ], dim=1)

        rays = self._allocate_rays(width * height)
        weight = 1.

        rays.src = src
        rays.dst = dst
        rays.wgt = weight
        rays.col = self._as_float_tensor(self.ray_col.asarray())

        return rays

    def _allocate_rays(self, n: int, **kwargs) -> Rays:
        return Rays.allocate(n, float_dtype=self.float_dtype, int_dtype=self.int_dtype, device=self.device, **kwargs)

    def _as_float_tensor(self, data, **kwargs) -> torch.Tensor:
        return torch.tensor(data, requires_grad=False, dtype=self.float_dtype, **kwargs).to(device=self.device)

    def _zeros(self, size, **kwargs) -> torch.Tensor:
        return torch.zeros(size, requires_grad=False, dtype=self.float_dtype, **kwargs).to(device=self.device)
