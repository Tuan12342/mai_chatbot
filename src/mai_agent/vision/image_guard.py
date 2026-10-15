from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
MAX_IMAGE_PIXELS = 25_000_000
MAX_OUTPUT_EDGE = 1600
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class InvalidSkinImage(ValueError):
    pass


def sanitize_skin_image(data: bytes, declared_mime_type: str, max_bytes: int) -> tuple[bytes, str]:
    """Xác thực, xoay đúng chiều, resize và mã hóa lại để loại metadata/EXIF."""
    if not data:
        raise InvalidSkinImage("Ảnh không được để trống.")
    if len(data) > max_bytes:
        raise InvalidSkinImage(f"Ảnh vượt quá giới hạn {max_bytes // (1024 * 1024)} MB.")
    if declared_mime_type not in SUPPORTED_IMAGE_TYPES:
        raise InvalidSkinImage("Chỉ hỗ trợ ảnh JPEG, PNG hoặc WebP.")

    try:
        with Image.open(BytesIO(data)) as source:
            source.verify()
        with Image.open(BytesIO(data)) as source:
            actual_format = source.format
            if actual_format != SUPPORTED_IMAGE_TYPES[declared_mime_type]:
                raise InvalidSkinImage("Định dạng thực tế của ảnh không khớp Content-Type.")
            image = ImageOps.exif_transpose(source)
            if image.width < 160 or image.height < 160:
                raise InvalidSkinImage("Ảnh quá nhỏ; mỗi chiều cần ít nhất 160 px.")
            image.thumbnail((MAX_OUTPUT_EDGE, MAX_OUTPUT_EDGE))
            if image.mode not in {"RGB", "L"}:
                background = Image.new("RGB", image.size, "white")
                if "A" in image.getbands():
                    background.paste(image, mask=image.getchannel("A"))
                else:
                    background.paste(image)
                image = background
            elif image.mode == "L":
                image = image.convert("RGB")

            output = BytesIO()
            image.save(output, format="JPEG", quality=88, optimize=True)
            return output.getvalue(), "image/jpeg"
    except InvalidSkinImage:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise InvalidSkinImage("File tải lên không phải ảnh hợp lệ hoặc ảnh đã bị lỗi.") from error
