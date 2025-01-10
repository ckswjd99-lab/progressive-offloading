import heapq
from collections import defaultdict
import numpy as np

def build_huffman_tree(data):
    """
    Builds a Huffman tree for the given data.

    Args:
        data (numpy.ndarray): Input data.

    Returns:
        dict: Huffman codes for each value in the data.
    """
    frequency = defaultdict(int)
    for value in data.ravel():
        frequency[value] += 1

    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huffman_codes = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    return {symbol: code for symbol, code in huffman_codes}

def huffman_encode(data, huffman_codes):
    """
    Encodes the data using Huffman codes.

    Args:
        data (numpy.ndarray): Input data.
        huffman_codes (dict): Huffman codes.

    Returns:
        tuple: Encoded data as binary and its original bitstring length.
    """
    bitstring = ''.join(huffman_codes[value] for value in data.ravel())
    bitstring_length = len(bitstring)
    encoded_data = int(bitstring, 2).to_bytes((bitstring_length + 7) // 8, byteorder='big')
    return encoded_data, bitstring_length

def huffman_decode(encoded_data, bitstring_length, huffman_codes, shape):
    """
    Decodes a Huffman encoded binary data back into an array.

    Args:
        encoded_data (bytes): Huffman encoded binary data.
        bitstring_length (int): Length of the original bitstring.
        huffman_codes (dict): Huffman codes.
        shape (tuple): Shape of the original data.

    Returns:
        numpy.ndarray: Decoded data.
    """
    reverse_codes = {code: symbol for symbol, code in huffman_codes.items()}
    bitstring = bin(int.from_bytes(encoded_data, byteorder='big'))[2:]
    bitstring = bitstring.zfill(bitstring_length)

    decoded_data = []
    current_code = ""
    for bit in bitstring:
        current_code += bit
        if current_code in reverse_codes:
            decoded_data.append(reverse_codes[current_code])
            current_code = ""
    return np.array(decoded_data, dtype=np.int16).reshape(shape)
