from io import BytesIO

import pytest
from PIL import Image

from mai_agent.vision.image_guard import InvalidSkinImage, sanitize_skin_image


def _jpeg_bytes(size: tuple[int, int] = (320, 240), *, with_exif: bool = False) -> bytes:
    image = Image.new("RGB", size, "#d9a18f")
    output = BytesIO()
    exif = Image.Exif()
    if with_exif:
        exif[0x010F] = "private-camera"
    image.save(output, format="JPEG", exif=exif)
    return output.getvalue()


def test_sanitize_reencodes_image_and_removes_exif() -> None:
    sanitized, mime_type = sanitize_skin_image(
        _jpeg_bytes(with_exif=True),
        "image/jpeg",
        8 * 1024 * 1024,
    )

    assert mime_type == "image/jpeg"
    with Image.open(BytesIO(sanitized)) as image:
        assert image.size == (320, 240)
        assert not image.getexif()


def test_rejects_mime_type_mismatch() -> None:
    with pytest.raises(InvalidSkinImage, match="không khớp"):
        sanitize_skin_image(_jpeg_bytes(), "image/png", 8 * 1024 * 1024)


def test_rejects_too_small_image() -> None:
    with pytest.raises(InvalidSkinImage, match="quá nhỏ"):
        sanitize_skin_image(_jpeg_bytes((100, 100)), "image/jpeg", 8 * 1024 * 1024)
