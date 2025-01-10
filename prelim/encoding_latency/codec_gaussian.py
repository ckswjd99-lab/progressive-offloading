import cv2
import numpy as np
import os
import time
from huffman import build_huffman_tree, huffman_encode, huffman_decode

def prod(val):
    val = list(val)
    res = 1
    for ele in val: 
        res *= ele 
    return res

def gaussian_pyramid_encode(image, levels):
    """
    Encodes an image into a Gaussian pyramid.

    Args:
        image (numpy.ndarray): Input image.
        levels (int): Number of levels in the pyramid.

    Returns:
        list: Gaussian pyramid (list of numpy arrays).
    """
    gaussian_pyramid = [image.astype(np.int16)]
    for i in range(levels - 1):
        image = cv2.pyrDown(image).astype(np.int16)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def gaussian_pyramid_decode(gaussian_pyramid, target_level, original_shape):
    """
    Decodes a Gaussian pyramid back into an image.

    Args:
        gaussian_pyramid (list): Gaussian pyramid (list of numpy arrays).
        target_level (int): The level to reconstruct to.
        original_shape (tuple): The original shape of the image.

    Returns:
        numpy.ndarray: Reconstructed image.
    """
    return gaussian_pyramid[target_level][:original_shape[0], :original_shape[1], :original_shape[2]]

# Example usage:
if __name__ == "__main__":
    image = cv2.imread("original.jpg")
    image = cv2.resize(image, (224, 224))
    original_shape = image.shape
    print(f"Original image {original_shape}: {prod(original_shape):,d} Bytes with {image.dtype}")

    os.makedirs("output", exist_ok=True)

    levels = 4
    start_time = time.time()
    gaussian_pyramid = gaussian_pyramid_encode(image, levels)
    print(f"Creating Gaussian Pyramid: {time.time() - start_time:.4f} sec")
    print([x.shape for x in gaussian_pyramid])

    # Directly create the lowest Gaussian layer
    for lev in range(levels):
        start_time = time.time()
        lowimage = image
        for _ in range(lev):
            lowimage = cv2.pyrDown(image)
        print(f"Creating Gaussian Layer {lev}: {time.time() - start_time:.4f} sec")

    total_compressed_size = 0
    for level, gaussian in reversed(list(enumerate(gaussian_pyramid))):
        cv2.imwrite(f"output/Gaussian_Level_{level}.png", gaussian.astype(np.uint8))

        start_time = time.time()
        huffman_codes = build_huffman_tree(gaussian)
        encoded_data, bitstring_length = huffman_encode(gaussian, huffman_codes)
        print(f"Encoding Gaussian Pyramid Level {level}: {time.time() - start_time:.4f} sec")

        compressed_size = len(encoded_data)
        total_compressed_size += compressed_size

        uncompressed_size = prod(gaussian.shape)
        print(f" >> Compressed size {gaussian.shape}, {gaussian.dtype}: {compressed_size:,d} bytes (uncomp. {uncompressed_size:,d} bytes, {compressed_size / uncompressed_size * 100:.2f}%)")

        decoded_layer = huffman_decode(encoded_data, bitstring_length, huffman_codes, gaussian.shape)
        assert np.array_equal(decoded_layer, gaussian), "Decoded layer does not match original!"

    print(f"Total compressed size: {total_compressed_size:,d} bytes")

    reconstructed_image = gaussian_pyramid_decode(gaussian_pyramid, 0, original_shape)

    cv2.imwrite("output/Original_Image.png", image)
    cv2.imwrite("output/Reconstructed_Image.png", reconstructed_image)

    assert np.allclose(reconstructed_image, image), "Reconstructed image does not match original!"
