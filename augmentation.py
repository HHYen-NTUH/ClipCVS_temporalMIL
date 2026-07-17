"""Utilities of data augmentation."""

from collections.abc import Callable
import enum
import torchvision.tv_tensors as tv_tensors
import torchvision.transforms.v2 as tvtrnsfrm


IMAGE_TRANSFORM = Callable[[tv_tensors.Image], tv_tensors.Image]


def zoom_augment(
    base_size: int = 480, ratio_range: tuple[float, float] = (1, 1)
) -> IMAGE_TRANSFORM:
    """
    Return transform of zooming augmentation.

    Argument
    --------
    ratio_range: Range of random resizing ratio
    """
    min_size = int(round(base_size * ratio_range[0]))
    max_size = int(round(base_size * ratio_range[1]))
    return tvtrnsfrm.RandomResize(min_size, max_size)


def color_jitter_augment(
    brightness_range: tuple[float, float] = (1, 1),
    contrast_range: tuple[float, float] = (1, 1),
    saturation_range: tuple[float, float] = (1, 1),
    hue_range: tuple[float, float] = (0, 0),
) -> IMAGE_TRANSFORM:
    """
    Return transform of color jittering augmentation.

    Arguments
    ---------
    brightness_range: Range of brightness augmentation
    contrast_range: Range of contrast augmentation
    saturation_range: Range of saturation augmentation
    hue_range: Range of hue augmentation
    """
    return tvtrnsfrm.ColorJitter(
        brightness=brightness_range,
        contrast=contrast_range,
        saturation=saturation_range,
        hue=hue_range,
    )


class AugmentFlag(enum.Flag):
    """Flags of applying augmentation."""

    NONE = 0
    ZOOM = enum.auto()
    COLOR = enum.auto()


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
