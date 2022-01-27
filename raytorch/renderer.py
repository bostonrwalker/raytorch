from raytorch.surface import Surface
from raytorch.texture import Texture
from raytorch.camera import Camera
from PIL import Image
from abc import ABC, abstractmethod
from typing import Iterable, Sequence


class Renderer(ABC):

    @abstractmethod
    def load_textures(self, textures: Sequence[Texture]):
        """
        Prepare the renderer to handle textures

        :param textures: Sequence of textures, linked through surface texture indices
        """
        pass

    @abstractmethod
    def set_fov(self, fov: float, width: int, height: int, **kwargs):
        """
        Fix the renderer FOV in eye space

        :param fov: Horizontal field of view in degrees
        :param width: Width of image to render in pixels
        :param height: Height of image to render in pixels
        """
        pass

    @abstractmethod
    def render(self, camera: Camera, world: Iterable[Surface], **kwargs) -> Image:
        """
        Render a scene

        :param camera: Camera representing viewpoint to render from
        :param world: Collection of all Surface objects present in world
        :return: Image
        """
        pass
