import timm
import time
import torch
import numpy as np
import multiprocessing as mp
from PIL import Image

from torchvision import transforms

import socket
import cv2

from networks import recv_bytes, recv_int, recv_ndarray, send_int, send_ndarray, sync_clock_server
from constants import *


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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def func_receive(server_port, queue_to_inference):
    socket_es = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_es.bind((SELF_IP, server_port))
    socket_es.listen(1)
    print(f"[func_receive] Listening on {server_port}")

    conn, addr = socket_es.accept()

    server_lagging, time_start = sync_clock_server(conn)
    print(f"[func_receive] Connection from {addr}, lagging {server_lagging * 1000:.6f} ms")

    while True:
        header = recv_int(conn)

        if header == HEADER_FINISH:
            print("[func_receive] Finish")
            queue_to_inference.put((None, None))
            break

        elif header == HEADER_IMAGE:
            level = recv_int(conn)
            image = recv_ndarray(conn)
            datasize = image.nbytes

            time_elapsed = time.time() - time_start - server_lagging
            print(f"[func_receive] Received image: level {level} {image.shape} ({datasize:,d} bytes) at {time_elapsed * 1000:.6f} ms")

            queue_to_inference.put((level, image))

        elif header == HEADER_RESULT:
            level = recv_int(conn)
            result = recv_ndarray(conn)

            print(f"[func_receive] Received result: level {level} {result.shape}")

        else:
            print(f"[func_receive] Invalid header: {header}")
            raise ValueError('Invalid header')
        
    conn.close()
    socket_es.close()


def func_inference(queue_from_es, queue_to_se, model_name=SERVER_MODELS[0]):
    model = timm.create_model(model_name, pretrained=True).to(DEVICE)

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"[func_inference] Model loaded: {model_name}")

    while True:
        level, image = queue_from_es.get()

        if level == None:
            print(f"[func_inference] Finish")
            queue_to_se.put(('Finish', None))
            break
        image = image.astype(np.uint8)
        # permute image to (B, C, H, W)
        # image = np.transpose(image, (2, 0, 1))
        # image = np.expand_dims(image, axis=0)
        print(f"[func_inference] Processing image: level {level} {image.shape}")
        image = Image.fromarray(image)
        image = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            result = model(image)
            result = result.cpu().numpy()

        print(f"[func_inference] Processed image: level {level} {result.shape}")
        queue_to_se.put(('Result', (level, result)))


def func_send(server_port, queue_from_inference):
    socket_se = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_se.bind((SELF_IP, server_port))
    socket_se.listen(1)
    print(f"[func_send] Listening on {server_port}")

    conn, addr = socket_se.accept()
    print(f"[func_send] Connection from {addr}")

    while True:
        received = queue_from_inference.get()
        header, data = received

        if header == 'Finish':
            send_int(conn, HEADER_FINISH)
            print(f"[func_send] Finish")
            break

        elif header == 'Result':
            level, result = data
            print(f"[func_send] Sending result: level {level} {result.shape}")

            # Send header
            send_int(conn, HEADER_RESULT)

            # Send level
            send_int(conn, level)

            # Send data
            result = result.astype(np.float32)
            send_ndarray(conn, result)

        else:
            raise ValueError('Invalid header')
        
    conn.close()
    socket_se.close()
        

def run_server(server_port_es, server_port_se):
    mp.set_start_method('spawn')

    queue_to_inference = mp.Queue()
    queue_from_inference = mp.Queue()

    p_es = mp.Process(target=func_receive, args=(server_port_es, queue_to_inference))
    p_infer = mp.Process(target=func_inference, args=(queue_to_inference, queue_from_inference))
    p_se = mp.Process(target=func_send, args=(server_port_se, queue_from_inference))

    p_es.start()
    p_infer.start()
    p_se.start()

    p_es.join()
    p_infer.join()
    p_se.join()

if __name__ == '__main__':
    run_server(SERVER_PORT_ES, SERVER_PORT_SE)
