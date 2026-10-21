"""图像处理与预处理模块。

职责：

- 面向桌面印刷体文字的 OCR 前处理（小字号增强、对比度提升）；
- 为原位渲染提供背景明暗判断，使译文遮罩自动适配深色/浅色界面。

自研内容：桌面场景预处理参数、背景明暗判定、文字主色估计。
"""

from __future__ import annotations

from dataclasses import dataclass

from ...utils.logger import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover
    import cv2
    import numpy as np

    _CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _CV_AVAILABLE = False


@dataclass
class RegionAppearance:
    """区域外观统计，用于译文样式自适应。"""

    #: 平均亮度，0-255
    mean_luminance: float
    #: 是否为深色背景（深色界面需要浅色译文与更深遮罩）
    is_dark: bool
    #: 估计的文字颜色（BGR），供渲染层参考
    text_color_bgr: tuple[int, int, int]
    #: 估计的背景颜色（BGR）
    background_color_bgr: tuple[int, int, int]


def is_available() -> bool:
    """返回图像处理依赖是否就绪。"""
    return _CV_AVAILABLE


def enhance_for_ocr(image, upscale: float = 1.0):
    """对捕获图像做 OCR 前处理。

    针对电脑界面印刷体文字的特点：

    - 小字号文字在缩放后更容易被检测到（``upscale > 1``）；
    - 轻微锐化可提升细笔画的边缘对比度；
    - CLAHE 局部对比度均衡，缓解浅灰字/深色背景的低对比问题。

    :param image: BGR 图像。
    :param upscale: 放大倍数，``1.0`` 表示不放大。
    :return: 处理后的图像；依赖缺失时原样返回。
    """
    if not _CV_AVAILABLE or image is None:
        return image

    result = image

    if upscale and abs(upscale - 1.0) > 1e-3:
        height, width = result.shape[:2]
        # 使用 INTER_CUBIC 在放大时保留笔画边缘，避免 INTER_LINEAR 造成的模糊。
        result = cv2.resize(
            result,
            (max(1, int(width * upscale)), max(1, int(height * upscale))),
            interpolation=cv2.INTER_CUBIC,
        )

    # 局部对比度均衡：在 LAB 亮度通道上做 CLAHE，避免色偏。
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    result = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

    # 轻度锐化：仅增强边缘，不过度放大噪点。
    blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.0)
    result = cv2.addWeighted(result, 1.4, blurred, -0.4, 0)

    return result


def downscale_boxes(
    boxes: list[tuple[int, int, int, int]],
    factor: float,
) -> list[tuple[int, int, int, int]]:
    """把在放大图上得到的文本框坐标还原回原图尺度。

    :param boxes: ``(x, y, w, h)`` 列表。
    :param factor: 之前使用的放大倍数。
    """
    if not factor or abs(factor - 1.0) < 1e-3:
        return boxes

    inverse = 1.0 / factor
    return [
        (
            round(x * inverse),
            round(y * inverse),
            max(1, round(w * inverse)),
            max(1, round(h * inverse)),
        )
        for x, y, w, h in boxes
    ]


def downscale_for_detection(image, long_side: int) -> tuple[object, float]:
    """把图像按长边缩到 ``long_side`` 以内，供 OCR 的**检测**阶段使用。

    检测耗时与输入像素数成正比，而屏幕上的字幕条常宽达 1600px，按原尺寸
    检测会在这一步吃掉上百毫秒。缩到 800px 长边后检测图像素只剩约 1/4。
    识别阶段仍从**原分辨率**图裁剪文本框，因此识别精度不受影响。

    :param image: BGR 图像。
    :param long_side: 目标长边像素数；``<= 0`` 表示不缩放。
    :return: ``(缩放后的图像, 缩放比例)``；比例为 ``1.0`` 时返回原图。
        不满足缩放条件（依赖缺失、尺寸已够小）时原样返回。
    """
    if not _CV_AVAILABLE or image is None or long_side <= 0:
        return image, 1.0

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= long_side:
        return image, 1.0

    ratio = long_side / float(longest)
    target_w = max(1, int(round(width * ratio)))
    target_h = max(1, int(round(height * ratio)))

    # 宽高必须按**同一比例**缩放。曾按 32 的整数倍取整尺寸，结果竖向上
    # 被多缩了一次（158px 高 → 64px，实际比例 2.47 而调用方按 2.0 换算），
    # 检测框整体上移，识别结果变成 "Wo don't hoyo muoh timo." 这类错字。
    # 网络下采样所需的 32 整数倍由库内部处理，这里不做。
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized, width / float(target_w)


def analyse_region(image, box: tuple[int, int, int, int] | None = None) -> RegionAppearance:
    """统计区域外观，用于让译文样式适配原文背景。

    :param image: BGR 图像（局部坐标系）。
    :param box: ``(x, y, w, h)`` 局部坐标；``None`` 表示整幅图像。
    :return: :class:`RegionAppearance`。
    """
    default = RegionAppearance(
        mean_luminance=128.0,
        is_dark=False,
        text_color_bgr=(0, 0, 0),
        background_color_bgr=(255, 255, 255),
    )
    if not _CV_AVAILABLE or image is None:
        return default

    patch = image
    if box is not None:
        x, y, w, h = box
        height, width = image.shape[:2]
        left, top = max(0, x), max(0, y)
        right, bottom = min(width, x + w), min(height, y + h)
        if right <= left or bottom <= top:
            return default
        patch = image[top:bottom, left:right]

    if patch.size == 0:
        return default

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mean_luminance = float(gray.mean())
    is_dark = mean_luminance < 110.0

    # 用 Otsu 把区域分为前景（文字）与背景两簇，再按簇的占比判断哪一簇是文字。
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground_ratio = float(np.count_nonzero(mask)) / mask.size
    # 文字通常占少数像素；前景占比超过一半说明极性相反。
    text_is_bright = foreground_ratio <= 0.5

    bright_pixels = patch[mask > 0]
    dark_pixels = patch[mask == 0]
    bright_mean = bright_pixels.mean(axis=0) if bright_pixels.size else np.array([255, 255, 255])
    dark_mean = dark_pixels.mean(axis=0) if dark_pixels.size else np.array([0, 0, 0])

    if text_is_bright:
        text_color = bright_mean
        background_color = dark_mean
    else:
        text_color = dark_mean
        background_color = bright_mean

    return RegionAppearance(
        mean_luminance=mean_luminance,
        is_dark=is_dark,
        text_color_bgr=tuple(int(v) for v in text_color),  # type: ignore[arg-type]
        background_color_bgr=tuple(int(v) for v in background_color),  # type: ignore[arg-type]
    )


def dominant_background_color(image, box: tuple[int, int, int, int] | None = None) -> tuple[int, int, int]:
    """返回区域主背景色（BGR），用于生成同色遮罩以减少视觉突兀。"""
    return analyse_region(image, box).background_color_bgr
