import numpy as np
from matplotlib.image import imread


def read_image(path) -> np.array:
    image = imread(path)
    if image.dtype == np.float32:
        image = image * 255
    return image.astype(np.uint8)
