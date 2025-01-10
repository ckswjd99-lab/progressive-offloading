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

def subsampling_encode(image, levels):
    """
    Encodes an image into a subsampling pyramid.

    Args:
        image (numpy.ndarray): Input image.
        levels (int): Number of levels in the pyramid.

    Returns:
        list: Subsampling pyramid (list of numpy arrays).
    """
    subsampling_pyramid = [image]
    for i in range(levels - 1):
        image = image[::2, ::2]
        subsampling_pyramid.append(image)

    return subsampling_pyramid

def subsampling_decode(subsampling_pyramid, target_level, original_shape):
    """
    Decodes a subsampling pyramid back into an image.

    Args:
        subsampling_pyramid (list): Subsampling pyramid (list of numpy arrays).
        target_level (int): The level to reconstruct to.
        original_shape (tuple): The original shape of the image.

    Returns:
        numpy.ndarray: Reconstructed image.
    """
    image = subsampling_pyramid[target_level]
    # Upsample until the image matches the original shape
    for i in range(target_level):
        image = np.repeat(image, 2, axis=0)
        image = np.repeat(image, 2, axis=1)

    return image[:original_shape[0], :original_shape[1], :original_shape[2]]

# Example usage:
if __name__ == "__main__":
    image = cv2.imread("original.jpg")
    image = cv2.resize(image, (224, 224))
    original_shape = image.shape
    print(f"Original image {original_shape}: {prod(original_shape):,d} Bytes with {image.dtype}")

    os.makedirs("output", exist_ok=True)

    levels = 4
    start_time = time.time()
    subsampling_pyramid = subsampling_encode(image, levels)
    print(f"Creating Subsampling Pyramid: {time.time() - start_time:.4f} sec")
    print([x.shape for x in subsampling_pyramid])

    total_compressed_size = 0
    for level, subsampled in reversed(list(enumerate(subsampling_pyramid))):
        cv2.imwrite(f"output/Subsampling_Level_{level}.png", subsampled.astype(np.uint8))

        start_time = time.time()
        huffman_codes = build_huffman_tree(subsampled)
        encoded_data, bitstring_length = huffman_encode(subsampled, huffman_codes)
        print(f"Encoding Subsampling Pyramid Level {level}: {time.time() - start_time:.4f} sec")

        compressed_size = len(encoded_data)
        total_compressed_size += compressed_size

        uncompressed_size = prod(subsampled.shape)
        print(f" >> Compressed size {subsampled.shape}, {subsampled.dtype}: {compressed_size:,d} bytes (uncomp. {uncompressed_size:,d} bytes, {compressed_size / uncompressed_size * 100:.2f}%)")

        decoded_layer = huffman_decode(encoded_data, bitstring_length, huffman_codes, subsampled.shape)
        assert np.array_equal(decoded_layer, subsampled), "Decoded layer does not match original!"

    print(f"Total compressed size: {total_compressed_size:,d} bytes")

    reconstructed_image = subsampling_decode(subsampling_pyramid, 0, original_shape)

    cv2.imwrite("output/Original_Image.png", image)
    cv2.imwrite("output/Reconstructed_Image.png", reconstructed_image)

    assert np.allclose(reconstructed_image, image), "Reconstructed image does not match original!"
