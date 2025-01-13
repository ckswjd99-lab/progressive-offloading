import timm
import torch
import numpy as np
import multiprocessing as mp
from PIL import Image

import socket
import cv2

from networks import receive_socket

SERVER_IP = '127.0.0.1'
SERVER_PORT_ES = 5000
SERVER_PORT_SE = 5001

NUM_REPEATS = 10

SERVER_MODELS = [
    'caformer_b36.sail_in22k_ft_in1k',
    'caformer_m36.sail_in22k_ft_in1k',
    'caformer_s36.sail_in22k_ft_in1k',

    'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k',
    'vit_large_patch14_clip_224.openai_ft_in12k_in1k',
    'vit_base_patch8_224.augreg2_in21k_ft_in1k',
    
    'convformer_b36.sail_in22k_ft_in1k',
    'convformer_m36.sail_in22k_ft_in1k',
    'convformer_s36.sail_in22k_ft_in1k',
    
    'deit3_huge_patch14_224.fb_in22k_ft_in1k',
    'deit3_large_patch16_224.fb_in22k_ft_in1k',
    'deit3_medium_patch16_224.fb_in22k_ft_in1k',
    'deit3_small_patch16_224.fb_in22k_ft_in1k',
]

HEADER_FINISH = 1000
HEADER_IMAGE = 1001
HEADER_RESULT = 1002


def func_dummy_receive(server_port):
    socket_dummy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_dummy.bind((SERVER_IP, server_port))
    socket_dummy.listen(1)
    print(f"[func_dummy_receive] Listening on {server_port}")

    conn, addr = socket_dummy.accept()
    print(f"[func_dummy_receive] Connection from {addr}")

    while True:
        received = receive_socket(conn, 4)
        header = int.from_bytes(received, byteorder='big')
        
        if header == HEADER_FINISH:
            print("[func_dummy_receive] Finish")
            # break

        elif header == HEADER_IMAGE:
            received = receive_socket(conn, 4)
            datasize = int.from_bytes(received, byteorder='big')

            received = receive_socket(conn, datasize)

            if datasize == 9408:
                image_shape = (1, 3, 28, 28)
            elif datasize == 37632:
                image_shape = (1, 3, 56, 56)
            elif datasize == 150528:
                image_shape = (1, 3, 112, 112)
            elif datasize == 602112:
                image_shape = (1, 3, 224, 224)
            else:
                raise ValueError('Invalid image size')

            image = np.frombuffer(received, dtype=np.float32).reshape(image_shape)
            print(f"[func_dummy_receive] Received image: {image.shape} ({datasize} bytes)")

        elif header == HEADER_RESULT:
            received = receive_socket(conn, 4)
            datasize = int.from_bytes(received, byteorder='big')

            received = receive_socket(conn, datasize)
            result = np.frombuffer(received, dtype=np.float32).reshape(1, 1000)
            print(f"[func_dummy_receive] Received result: {result.shape}")

        else:
            print(f"[func_dummy_receive] Invalid header: {header}")
            raise ValueError('Invalid header')
        
    conn.close()
    socket_dummy.close()

def run_dummy_receive(server_port_es, server_port_se):
    p_es = mp.Process(target=func_dummy_receive, args=(server_port_es,))
    p_se = mp.Process(target=func_dummy_receive, args=(server_port_se,))

    p_es.start()
    p_se.start()

    p_es.join()
    p_se.join()

if __name__ == '__main__':
    run_dummy_receive(SERVER_PORT_ES, SERVER_PORT_SE)
