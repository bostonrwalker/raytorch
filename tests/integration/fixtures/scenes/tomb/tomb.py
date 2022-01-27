from pathlib import Path

from raytorch.camera import Camera
from raytorch.core import Vector
from raytorch.scene import Scene
from raytorch.surface.parallelogram import Parallelogram
from raytorch.surface.sphere import Sphere
from raytorch.texture.image_texture import ImageTexture
from raytorch.torch_raytracer import RayTracer

from tests.integration.fixtures.scenes.tomb.textures.glass import Glass
from tests.integration.fixtures.scenes.tomb.textures.metal import Metal


test_fixture_path = Path(__file__).parent

texture_dict = {
    "wall": ImageTexture(test_fixture_path / "textures/hieroglyphics/diffuse.jpg", use_tensors=True),
    "floor": ImageTexture(test_fixture_path / "textures/stone_path/diffuse.tiff", use_tensors=True),
    "air_to_glass": Glass(normal_map_path=(test_fixture_path / "textures/energy_pole/normal.tif")),
    "glass_to_air": Glass(normal_map_path=(test_fixture_path / "textures/energy_pole/normal.tif"), inverted=True),
    "metal": Metal(normal_map_path=(test_fixture_path / "textures/energy_pole/normal.tif"))
}

# Textures are mapped using integer indices
texture_indices = {k: i for i, k in enumerate(texture_dict)}
textures = list(texture_dict.values())

surfaces = [
    Parallelogram(Vector(1., 0., 0.), Vector(0., 0., 1.), Vector(0., 1., 0.), texture=texture_indices['wall']),
    Parallelogram(Vector(1., 0., 1.), Vector(-1., 0., 0.), Vector(0., 1., 0.), texture=texture_indices['wall']),
    Parallelogram(Vector(0., 0., 1.), Vector(0., 0., -1.), Vector(0., 1., 0.), texture=texture_indices['wall']),
    Parallelogram(Vector(0., 0., 0.), Vector(1., 0., 0.), Vector(0., 1., 0.), texture=texture_indices['wall']),
    Parallelogram(Vector(0., 0., 0.), Vector(0., 0., 1.), Vector(1., 0., 0.), texture=texture_indices['floor']),
    Parallelogram(Vector(0., 1., 0.), Vector(1., 0., 0.), Vector(0., 0., 1.), texture=texture_indices['floor']),
    Sphere(Vector(.3, .25, .5), .25, texture=texture_indices['air_to_glass']),
    Sphere(Vector(.3, .25, .5), .25, texture=texture_indices['glass_to_air'], inverted=True),
    Sphere(Vector(.8, .15, .3), .15, texture=texture_indices['metal'])
]

camera = Camera.upright_camera(Vector(0.02, .75, 0.02), Vector(.75, 0., .75))

renderer = RayTracer(ray_offset_len=5e-2)
renderer.set_fov(fov=75, width=1280, height=720)

Tomb = Scene(textures, surfaces, camera, renderer)
