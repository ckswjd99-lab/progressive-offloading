import timm
import time
import torch
import socket

import numpy as np
import multiprocessing as mp

import cv2
from PIL import Image

from samplers import sample_gaussian, sample_selection, sample_average


SERVER_IP = '127.0.0.1'
SERVER_PORT_ES = 5000
SERVER_PORT_SE = 5001

NUM_REPEATS = 10

SAMPLE_IMAGE_PATH = './data/input.jpg'

EDGE_MODEL_NAMES = [
    'efficientvit_b0.r224_in1k',
    'repghostnet_080.in1k',
    'tf_mobilenetv3_large_075.in1k',
    'mobilenetv3_small_050.lamb_in1k',
    'efficientvit_m1.r224_in1k',
    'vgg11_bn.tv_in1k',
]

HEADER_FINISH = 1000
HEADER_IMAGE = 1001
HEADER_RESULT = 1002

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


def func_inference(queue_from_main, queue_to_offload, model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    print(f"[func_inference] Loaded model {model_name}")

    image = queue_from_main.get()   # Pillow Image
    image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()  # torch.Tensor(1, 3, 224, 224)
    print(f"[func_inference] Received image: {image.shape}")

    output = model(image)
    output = output.detach().numpy() # np.ndarray(1, 1000)
    queue_to_offload.put(('Result', output))
    print(f"[func_inference] Inference done")

def func_sample(queue_from_main, queue_to_offload, sampler='average'):
    sample_func = {
        'gaussian': sample_gaussian,
        'selection': sample_selection,
        'average': sample_average,
    }[sampler]
        
    image = queue_from_main.get()   # np.ndarray(3, 224, 224)
    print(f"[func_sample] Received image: {image.shape}")

    for level in range(3, -1, -1):
        image_sampled = sample_func(image, level)
        queue_to_offload.put(('Image', image_sampled))
        print(f"[func_sample] Sampled image: {image_sampled.shape}")

def func_offload(queue_from_procs, server_ip, server_port, sampler='average'):
    socket_to_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_to_server.connect((server_ip, server_port))
    print(f"[func_offload] Connected to server {server_ip}:{server_port}")

    while True:
        received = queue_from_procs.get()
        header, data = received

        if header == 'Finish':
            print(f"[func_offload] Finish")
            break

        elif header == 'Image':
            # Send tensor
            data = data.astype(np.float32)
            bytedata = data.tobytes()
            datasize = len(bytedata).to_bytes(4, 'big')
            print(f"[func_offload] Sending image: {data.shape} ({len(bytedata)} bytes)")

            socket_to_server.send(HEADER_IMAGE.to_bytes(4, 'big'))
            socket_to_server.send(datasize)
            socket_to_server.send(bytedata)

        elif header == 'Result':
            print(f"[func_offload] Sending result: {data.shape}")
            # Send result
            bytedata = data.tobytes()
            datasize = len(bytedata).to_bytes(4, 'big')
            # socket_to_server.send(HEADER_RESULT.to_bytes(4, 'big'))
            # socket_to_server.send(datasize)
            # socket_to_server.send(bytedata)

        else:
            raise ValueError('Invalid header')


def func_receive(queue_to_main, server_ip, server_port):
    socket_from_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_from_server.connect((server_ip, server_port))
    print(f"[func_receive] Connected to server {server_ip}:{server_port}")

    while True:
        received = socket_from_server.recv(4)
        header = int.from_bytes(received, 'big')

        if header == HEADER_FINISH:
            queue_to_main.send(None)
            break

        elif header == HEADER_RESULT:
            received = socket_from_server.recv(4)
            datasize = int.from_bytes(received, 'big')
            received = socket_from_server.recv(datasize)
            result = torch.tensor(np.frombuffer(received, dtype=np.float32).reshape(1, 1000))
            queue_to_main.send(result)

        else:
            raise ValueError('Invalid header')
        

def run_offload(model_name=EDGE_MODEL_NAMES[-1]):
    image = image_preprocess(Image.open(SAMPLE_IMAGE_PATH))
    print(f"Image shape: {image.shape}")

    queue_to_inference = mp.Queue()
    queue_to_sample = mp.Queue()
    queue_to_offload = mp.Queue()
    queue_from_receiver = mp.Queue()

    proc_inference = mp.Process(target=func_inference, args=(queue_to_inference, queue_to_sample, model_name))
    proc_sample = mp.Process(target=func_sample, args=(queue_to_sample, queue_to_offload))
    proc_offload = mp.Process(target=func_offload, args=(queue_to_offload, SERVER_IP, SERVER_PORT_ES))
    proc_receive = mp.Process(target=func_receive, args=(queue_from_receiver, SERVER_IP, SERVER_PORT_SE))

    proc_inference.start()
    proc_sample.start()
    proc_offload.start()
    proc_receive.start()

    time.sleep(1)

    queue_to_inference.put(image)
    queue_to_sample.put(image)

    while True:
        received = queue_from_receiver.get()
        if received is None:
            break

        print(received)

    proc_inference.join()
    proc_sample.join()
    proc_offload.join()
    proc_receive.join()


if __name__ == '__main__':
    for _ in range(NUM_REPEATS):
        run_offload()