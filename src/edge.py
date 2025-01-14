import timm
import time
import torch
import socket

import numpy as np
import multiprocessing as mp

import cv2
from PIL import Image

from samplers import sample_gaussian, sample_selection, sample_average
from networks import send_int, send_float, send_ndarray, recv_int, recv_float, recv_ndarray, sync_clock_edge
from constants import *


SAMPLE_IMAGE_PATH = './data/input.jpg'

EDGE_MODEL_NAMES = [
    'efficientvit_b0.r224_in1k',
    'repghostnet_080.in1k',
    'tf_mobilenetv3_large_075.in1k',
    'mobilenetv3_small_050.lamb_in1k',
    'efficientvit_m1.r224_in1k',
    'vgg11_bn.tv_in1k',
]

def image_preprocess(image):
    if image.height < image.width:
        new_height = 256
        new_width = int(image.width * 256 / image.height)
    else:
        new_width = 256
        new_height = int(image.height * 256 / image.width)
    image = cv2.resize(np.array(image), (new_width, new_height), interpolation=cv2.INTER_AREA)

    left = (new_width - 224) // 2
    top = (new_height - 224) // 2
    right = left + 224
    bottom = top + 224
    image = image[top:bottom, left:right]

    return image


def func_inference(queue_from_main, queue_to_offload, queue_to_main, model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    print(f"[func_inference] Loaded model {model_name}")

    while True:
        data = queue_from_main.get()   # Pillow Image
        header, image = data

        if header == 'Finish':
            print(f"[func_inference] Finish")
            break

        elif header == 'Clock':
            server_lagging, time_start = image
            print(f"[func_inference] Sync clock: {server_lagging * 1000:.6f} ms")

        else:
            image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()  # torch.Tensor(1, 3, 224, 224)

            timestamp = time.time() - time_start
            print(f"[func_inference] Received image: {image.shape} at {timestamp * 1000:.6f} ms")

            output = model(image)
            output = output.detach().numpy() # np.ndarray(1, 1000)
            queue_to_main.put(('Result', (-1, output)))
        

def func_sample(queue_from_main, queue_to_offload, sampler='average'):
    sample_func = {
        'gaussian': sample_gaussian,
        'selection': sample_selection,
        'average': sample_average,
    }[sampler]
    
    while True:
        data = queue_from_main.get()   # np.ndarray(3, 224, 224)
        header, image = data
        
        if header == 'Finish':
            print(f"[func_sample] Finish")
            break
        
        elif header == 'Clock':
            server_lagging, time_start = image
            print(f"[func_sample] Sync clock: {server_lagging * 1000:.6f} ms")

        else:
            print(f"[func_sample] Received image: {image.shape}")

            for level in range(3, -1, -1):
                image_sampled = sample_func(image, level)
                queue_to_offload.put(('Image', (level, image_sampled)))

                timestamp = time.time() - time_start
                print(f"[func_sample] Sampled image: {image_sampled.shape} at {timestamp * 1000:.6f} ms")
    

def func_offload(queue_from_procs, server_ip, server_port, sampler='average'):
    socket_to_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # try until connected
    while True:
        try:
            socket_to_server.connect((server_ip, server_port))
            break
        except:
            pass

    while True:
        received = queue_from_procs.get()
        header, data = received

        if header == 'Finish':
            send_int(socket_to_server, HEADER_FINISH)
            print(f"[func_offload] Finish")
            break

        elif header == 'Clock':
            server_lagging, time_start = data

            send_int(socket_to_server, HEADER_CLOCK)
            send_float(socket_to_server, server_lagging)
            send_float(socket_to_server, time_start)

            print(f"[func_offload] Sync clock: {server_lagging * 1000:.6f} ms")

        elif header == 'Image':
            level, image = data
            image = image.astype(np.float32)
            image_size = image.size

            time_elapsed = time.time() - time_start
            print(f"[func_offload] Sending image: {image.shape} ({image_size:,d} bytes) at {time_elapsed * 1000:.6f} ms")

            # Send header
            send_int(socket_to_server, HEADER_IMAGE)
            
            # Send level
            send_int(socket_to_server, level)

            # Send data
            send_ndarray(socket_to_server, image)

        elif header == 'Result':
            level, result = data
            print(f"[func_offload] Sending result: {result.shape}")
            
            # Send header
            send_int(socket_to_server, HEADER_RESULT)

            # Send level
            send_int(socket_to_server, level)

            # Send data
            result = result.astype(np.float32)
            send_ndarray(socket_to_server, result)

        else:
            raise ValueError('Invalid header')


def func_receive(queue_to_main, server_ip, server_port):
    socket_from_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # try until connected
    while True:
        try:
            socket_from_server.connect((server_ip, server_port))
            break
        except:
            pass

    print(f"[func_receive] Connected to server {server_ip}:{server_port}")

    while True:
        received = socket_from_server.recv(4)
        header = int.from_bytes(received, 'big')

        if header == HEADER_FINISH:
            queue_to_main.put(('Finish', None))
            break

        elif header == HEADER_CLOCK:
            server_lagging = recv_float(socket_from_server)
            time_start = recv_float(socket_from_server)
            print(f"[func_receive] Sync clock: {server_lagging * 1000:.6f} ms")

        elif header == HEADER_RESULT:
            level = recv_int(socket_from_server)
            result = recv_ndarray(socket_from_server)
            print(f"[func_receive] Received result: level {level} {result.shape}")
            queue_to_main.put(('Result', (level, result)))

        else:
            raise ValueError('Invalid header')
        

def run_offload(model_name=EDGE_MODEL_NAMES[0], num_repeat=NUM_REPEATS, meta_port=SERVER_PORT_META):
    mp.set_start_method('spawn')

    # Meta socket
    socket_meta = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_meta.connect((SERVER_IP, meta_port))

    image = image_preprocess(Image.open(SAMPLE_IMAGE_PATH))
    print(f"Image shape: {image.shape}")

    # Create processes
    queue_to_inference = mp.Queue()
    queue_to_sample = mp.Queue()
    queue_to_offload = mp.Queue()
    queue_to_main = mp.Queue()

    proc_inference = mp.Process(target=func_inference, args=(queue_to_inference, queue_to_offload, queue_to_main, model_name))
    proc_sample = mp.Process(target=func_sample, args=(queue_to_sample, queue_to_offload))
    proc_offload = mp.Process(target=func_offload, args=(queue_to_offload, SERVER_IP, SERVER_PORT_ES))
    proc_receive = mp.Process(target=func_receive, args=(queue_to_main, SERVER_IP, SERVER_PORT_SE))

    proc_inference.start()
    proc_sample.start()
    proc_offload.start()
    proc_receive.start()

    time.sleep(1)

    inference_results = []

    for i in range(NUM_REPEATS):
        print(f"[run_offload] Offloading iter: {i+1}")

        # Sync clock
        server_lagging, time_start = sync_clock_edge(socket_meta)
        print(f"[run_offload] Connected to server {SERVER_IP}:{meta_port}, lagging {server_lagging * 1000:.6f} ms")

        queue_to_inference.put(('Clock', (server_lagging, time_start)))
        queue_to_sample.put(('Clock', (server_lagging, time_start)))
        queue_to_offload.put(('Clock', (server_lagging, time_start)))

        # Start offloading
        timestamp = time.time() - time_start
        print(f"[run_offload] Loaded image {image.shape} at {timestamp * 1000:.6f} ms")
        queue_to_inference.put(('Image', image))
        queue_to_sample.put(('Image', image))

        # Wait for results
        while True:
            received = queue_to_main.get()
            header, data = received

            if header == 'Finish':
                print(f"Finish")
                break
            elif header == 'Result':
                level, result = data
                inference_results.append((level, result))

                timestamp = time.time() - time_start
                print(f"[run_offload] Received result: level {level} {result.shape} at {timestamp * 1000:.6f} ms")

                if len(inference_results) == 5:
                    inference_results = []
                    queue_to_main.put(('Finish', None))
            else:
                raise ValueError('Invalid header')
            
        time.sleep(1)
    
    queue_to_offload.put(('Finish', None))
    queue_to_inference.put(('Finish', None))
    queue_to_sample.put(('Finish', None))
        

    proc_inference.join()
    proc_sample.join()
    proc_offload.join()
    proc_receive.join()


if __name__ == '__main__':
    run_offload()