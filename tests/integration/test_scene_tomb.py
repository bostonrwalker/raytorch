from typing import Optional

import pytest
from math import floor
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image, ImageChops

from raytorch.scene import Scene

from tests.integration.fixtures.scenes.tomb.tomb import Tomb


test_scenes_path = Path(__file__).parent / "fixtures" / "scenes"


def images_are_equal(image, expected_image, tolerance: Optional[int] = None) -> bool:
    """
    Compare two images for equality. Encode difference as jpg and assert that encoded size of difference is smaller
    than tolerance in bytes.
    """
    tolerance = tolerance or floor(0.02 * image.size[0] * image.size[1])

    diff = ImageChops.difference(image, expected_image)
    with TemporaryDirectory() as tmp_dir:
        jpg = Path(tmp_dir) / "diff.jpg"
        diff.save(jpg)
        return jpg.stat().st_size <= tolerance


@pytest.mark.parametrize(("scene", "expected_output"), [
    (Tomb, Image.open(test_scenes_path / "tomb" / "sample_output" / "tomb.png").convert("RGB")),
])
def test_scenes(scene: Scene, expected_output: Image):
    """ """
    output = scene.render()
    assert images_are_equal(output, expected_output)
