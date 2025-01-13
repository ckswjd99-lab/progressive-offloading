import cv2
import numpy as np

def sample_gaussian(image, num_downsample):
    """
    Downsample image using Gaussian pyramid.

    Args:
        image (numpy.ndarray): Input image.
        num_downsample (int): Number of downsampling levels.

    Returns:
        image (numpy.ndarray): Downsampled image.
    """
    for i in range(num_downsample):
        image = cv2.pyrDown(image)
    return image

def sample_selection(image, levels):
    """
    Encodes an image into a subsampling pyramid.

    Args:
        image (numpy.ndarray): Input image.
        levels (int): Number of levels in the pyramid.

    Returns:
        image (numpy.ndarray): Downsampled image.
    """
    image = image[::(2 ** levels), ::(2 ** levels)]
    return image

def sample_average(image, levels):
    """
    Encodes an image into an average subsampling pyramid.

    Args:
        image (numpy.ndarray): Input image.
        levels (int): Number of levels in the pyramid.

    Returns:
        image (numpy.ndarray): Downsampled image.
    """
    image = cv2.resize(image, (image.shape[1] // (2 ** levels), image.shape[0] // (2 ** levels)), interpolation=cv2.INTER_AREA)
    return image

