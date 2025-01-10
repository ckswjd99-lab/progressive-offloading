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

def laplacian_pyramid_encode(image, levels):
    """
    Encodes an image into a Laplacian pyramid.

    Args:
        image (numpy.ndarray): Input image.
        levels (int): Number of levels in the pyramid.

    Returns:
        list: Laplacian pyramid (list of numpy arrays).
    """
    gaussian_pyramid = [image.astype(np.int16)]
    for i in range(levels - 1):
        image = cv2.pyrDown(image).astype(np.int16)
        gaussian_pyramid.append(image)

    laplacian_pyramid = []
    for i in range(levels - 1, 0, -1):
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size).astype(np.int16)
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], expanded)
        laplacian_pyramid.append(laplacian)
        
    laplacian_pyramid = [gaussian_pyramid[-1]] + laplacian_pyramid

    return gaussian_pyramid, laplacian_pyramid

def laplacian_pyramid_decode(laplacian_pyramid, target_level):
    """
    Decodes a Laplacian pyramid back into an image.

    Args:
        laplacian_pyramid (list): Laplacian pyramid (list of numpy arrays).
        target_level (int): The level to reconstruct to.

    Returns:
        numpy.ndarray: Reconstructed image.
    """
    image = laplacian_pyramid[0]  # Start with the smallest Gaussian level
    for i in range(1, len(laplacian_pyramid) - target_level):
        size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        image = cv2.pyrUp(image, dstsize=size).astype(np.int16)
        image = cv2.add(image, laplacian_pyramid[i])

    return image

# Example usage:
if __name__ == "__main__":
    # Load an image
    image = cv2.imread("original.jpg")
    image = cv2.resize(image, (512, 512))  # Resize for simplicity
    original_shape = image.shape
    print(f"Original image {original_shape}: {prod(original_shape):,d} Bytes with {image.dtype}")

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Encode image into Laplacian pyramid
    levels = 4
    start_time = time.time()
    gaussian_pyramid, laplacian_pyramid = laplacian_pyramid_encode(image, levels)
    print(f"Creating Laplacian Pyramid: {time.time() - start_time:.4f} sec")
    

    # Save intermediate Laplacian layers
    total_compressed_size = 0
    for i, laplacian in enumerate(laplacian_pyramid):  # Exclude the smallest Gaussian layer
        level = levels - i - 1
        if i == 0:
            cv2.imwrite(f"output/Laplacian_Level_{level}.png", laplacian.astype(np.uint8))
        else:
            cv2.imwrite(f"output/Laplacian_Level_{level}.png", (laplacian + 128).astype(np.uint8))

        # Huffman encode each Laplacian layer
        start_time = time.time()
        huffman_codes = build_huffman_tree(laplacian)
        encoded_data, bitstring_length = huffman_encode(laplacian, huffman_codes)
        print(f"Encoding Laplacian Pyramid Level {level}: {time.time() - start_time:.4f} sec")

        # Calculate and print compressed size
        compressed_size = len(encoded_data)
        total_compressed_size += compressed_size

        uncompressed_size = prod(laplacian.shape)
        print(f" >> Compressed size {laplacian.shape}, {laplacian.dtype}: {compressed_size:,d} bytes (uncomp. {uncompressed_size:,d} bytes, {compressed_size / uncompressed_size * 100:.2f}%)")

        # Decode the encoded data for verification
        decoded_layer = huffman_decode(encoded_data, bitstring_length, huffman_codes, laplacian.shape)
        assert np.array_equal(decoded_layer, laplacian), "Decoded layer does not match original!"

    print(f"Total compressed size: {total_compressed_size:,d} bytes")

    # Decode Laplacian pyramid back to image
    for i in range(levels):
        level = levels - i - 1
        reconstructed_image = laplacian_pyramid_decode(laplacian_pyramid, level)
        
        # Upscale into original scale
        image = reconstructed_image
        for _ in range(level):
            size = (image.shape[0] * 2, image.shape[1] * 2)
            image = cv2.pyrUp(image, dstsize=size).astype(np.int16)

        cv2.imwrite(f"output/Recon_Level_{level}.png", image)


    # Save original and reconstructed images
    cv2.imwrite("output/Original_Image.png", image)
    cv2.imwrite("output/Reconstructed_Image.png", reconstructed_image)

    # Verify that the reconstructed image matches the original
    assert np.allclose(reconstructed_image, image), "Reconstructed image does not match original!"

