import numpy as np


def encode_shape_and_image(img: np.ndarray) -> np.ndarray:
    # encodes the shape and the actual data into one flat uint8 array by using the first 15 bytes to encode the shape
    assert img.dtype == np.uint8 and img.ndim == 3, "requires a uint8 image"
    res = np.ndarray(shape=(15 + img.nbytes,), dtype=img.dtype)
    for i, dim in enumerate(img.shape):
        for j, digit in enumerate(f"{dim:0>5d}"):
            res[i*5+j] = int(digit)
    res[15:] = img.flatten()[:]
    return res


def decode_shape_and_image(shpimg: np.ndarray) -> np.ndarray:
    # decodes a flat uint8 array into an image of correct shape by parsing the first 15 bytes assuming they encode the shape
    assert shpimg.dtype == np.uint8 and shpimg.ndim == 1, "requires a flat uint8 representations of an image"
    shp = (
        int(''.join([str(i) for i in shpimg[:5]])),
        int(''.join([str(i) for i in shpimg[5:10]])),
        int(''.join([str(i) for i in shpimg[10:15]]))
    )
    return shpimg[15:].reshape(shp)
