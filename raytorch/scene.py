from typing import Iterable

from PIL import Image

from raytorch.texture import Texture
from raytorch.surface import Surface
from raytorch.camera import Camera
from raytorch.renderer import Renderer


class Scene:

    def __init__(self, textures: Iterable[Texture], surfaces: Iterable[Surface], camera: Camera, renderer: Renderer):
        """
        A scene to be rendered. Convenience class for completely defining the rendering process for a scene.

        :param textures: Textures to be loaded/rendered onto surfaces
        :param surfaces: Surfaces to be rendered
        :param camera: Point of view to render from
        :param renderer: Renderer
        """
        self.textures = textures
        self.surfaces = surfaces
        self.camera = camera
        self.renderer = renderer

    def render(self) -> Image:
        """ Render scene and return an image """
        for texture in self.textures:
            texture.load()
        self.renderer.load_textures(list(self.textures))
        return self.renderer.render(self.camera, self.surfaces)
