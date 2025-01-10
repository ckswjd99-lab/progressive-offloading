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

def average_pyramid_encode(image, levels):
    """
    Encodes an image into an average subsampling pyramid.

    Args:
        image (numpy.ndarray): Input image.
        levels (int): Number of levels in the pyramid.

    Returns:
        list: Average subsampling pyramid (list of numpy arrays).
    """
    average_pyramid = [image.astype(np.int16)]
    for i in range(levels - 1):
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA).astype(np.int16)
        average_pyramid.append(image)
    return average_pyramid

def average_pyramid_decode(average_pyramid, target_level, original_shape):
    """
    Decodes an average subsampling pyramid back into an image.

    Args:
        average_pyramid (list): Average subsampling pyramid (list of numpy arrays).
        target_level (int): The level to reconstruct to.
        original_shape (tuple): The original shape of the image.

    Returns:
        numpy.ndarray: Reconstructed image.
    """
    image = average_pyramid[target_level]
    for i in range(target_level):
        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_LINEAR).astype(np.int16)
    return image[:original_shape[0], :original_shape[1], :original_shape[2]]

# Example usage:
if __name__ == "__main__":
    image = cv2.imread("original.jpg")
    image = cv2.resize(image, (512, 512))
    original_shape = image.shape
    print(f"Original image {original_shape}: {prod(original_shape):,d} Bytes with {image.dtype}")

    os.makedirs("output", exist_ok=True)

    levels = 4
    start_time = time.time()
    average_pyramid = average_pyramid_encode(image, levels)
    print(f"Creating Average Subsampling Pyramid: {time.time() - start_time:.4f} sec")
    print([x.shape for x in average_pyramid])

    total_compressed_size = 0
    for level, average in reversed(list(enumerate(average_pyramid))):
        cv2.imwrite(f"output/Average_Level_{level}.png", average.astype(np.uint8))

        start_time = time.time()
        huffman_codes = build_huffman_tree(average)
        encoded_data, bitstring_length = huffman_encode(average, huffman_codes)
        print(f"Encoding Average Subsampling Pyramid Level {level}: {time.time() - start_time:.4f} sec")

        compressed_size = len(encoded_data)
        total_compressed_size += compressed_size

        uncompressed_size = prod(average.shape)
        print(f" >> Compressed size {average.shape}, {average.dtype}: {compressed_size:,d} bytes (uncomp. {uncompressed_size:,d} bytes, {compressed_size / uncompressed_size * 100:.2f}%)")

        decoded_layer = huffman_decode(encoded_data, bitstring_length, huffman_codes, average.shape)
        assert np.array_equal(decoded_layer, average), "Decoded layer does not match original!"

    print(f"Total compressed size: {total_compressed_size:,d} bytes")

    reconstructed_image = average_pyramid_decode(average_pyramid, 0, original_shape)

    cv2.imwrite("output/Original_Image.png", image)
    cv2.imwrite("output/Reconstructed_Image.png", reconstructed_image)

    assert np.allclose(reconstructed_image, image), "Reconstructed image does not match original!"
