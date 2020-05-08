import os
NUM_THREADS = 1
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
os.environ['MKL_NUM_THREADS'] = str(NUM_THREADS)
from raytorch.core import Vector
from raytorch.surface import Parallelogram, Sphere
from raytorch.texture import ImageTexture, Glass, Metal
from raytorch import Camera, TorchRayTracer as RayTracer
from datetime import datetime


textures = {
    'wall': ImageTexture('textures/TexturesCom_OrnamentsEgyptian0017_1_M/diffuse.jpg', use_tensors=True),
    'floor': ImageTexture('textures/TexturesCom_MedievalFloor9Path_1K/diffuse.tiff', use_tensors=True),
    'air_to_glass': Glass(normal_map_path='textures/TexturesCom_Various_EnergyPole_2K/normal.tif'),
    'glass_to_air': Glass(normal_map_path='textures/TexturesCom_Various_EnergyPole_2K/normal.tif', inverted=True),
    'metal': Metal(normal_map_path='textures/TexturesCom_Various_EnergyPole_2K/normal.tif')
}


def _texture(s):
    return list(textures.keys()).index(s)


world = [
    Parallelogram(Vector(1., 0., 0.), Vector(0., 0., 1.), Vector(0., 1., 0.), texture=_texture('wall')),
    Parallelogram(Vector(1., 0., 1.), Vector(-1., 0., 0.), Vector(0., 1., 0.), texture=_texture('wall')),
    Parallelogram(Vector(0., 0., 1.), Vector(0., 0., -1.), Vector(0., 1., 0.), texture=_texture('wall')),
    Parallelogram(Vector(0., 0., 0.), Vector(1., 0., 0.), Vector(0., 1., 0.), texture=_texture('wall')),
    Parallelogram(Vector(0., 0., 0.), Vector(0., 0., 1.), Vector(1., 0., 0.), texture=_texture('floor')),
    Parallelogram(Vector(0., 1., 0.), Vector(1., 0., 0.), Vector(0., 0., 1.), texture=_texture('floor')),
    Sphere(Vector(.3, .25, .5), .25, texture=_texture('air_to_glass')),
    Sphere(Vector(.3, .25, .5), .25, texture=_texture('glass_to_air'), inverted=True),
    Sphere(Vector(.8, .15, .3), .15, texture=_texture('metal'))
]

camera = Camera.upright_camera(Vector(0.02, .75, 0.02), Vector(.75, 0., .75))

renderer = RayTracer(ray_offset_len=5e-2)

width = 1280
height = 720
fov = 75

output = 'tomb.png'

if __name__ == '__main__':

    print('Loading...')

    for texture in textures.values():
        texture.load()

    t1 = datetime.now()
    renderer.load_textures(list(textures.values()))
    renderer.set_fov(fov=fov, width=width, height=height)
    t2 = datetime.now()

    print(f'Elapsed time: {t2 - t1}')

    print('Rendering...')

    t1 = datetime.now()
    image = renderer.render(camera, world)
    t2 = datetime.now()

    print(f'Elapsed time: {t2 - t1}')

    print('Done.')

    if output is None:
        image.show()
    else:
        image.save(f'output/{output}')
