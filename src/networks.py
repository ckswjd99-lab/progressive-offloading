import numpy as np
import time
import struct

from constants import *


def recv_bytes(sock, bytes_to_receive):
    received = b''
    while len(received) < bytes_to_receive:
        received += sock.recv(bytes_to_receive - len(received))
    return received


def sync_clock_edge(socket_to_server):
    num_repeat = 10
    server_lagging = 0

    for _ in range(num_repeat):
        time_edge = time.time()
        send_float(socket_to_server, time_edge)
        time_server = recv_float(socket_to_server)
        time_round = time.time()
        server_lagging += time_server - (time_round + time_edge) / 2

        # print(f"Edge: {time_edge:.6f}, Server: {time_server:.6f}, Round: {time_round:.6f}")

    server_lagging /= num_repeat

    # send lagging time to server
    send_float(socket_to_server, server_lagging)
    
    # send start time to server
    time_start = time.time()
    send_float(socket_to_server, time_start)

    return server_lagging, time_start

def sync_clock_server(socket_from_edge):
    num_repeat = 10
    server_lagging = 0

    for _ in range(num_repeat):
        time_edge = recv_float(socket_from_edge)
        time_server = time.time()
        send_float(socket_from_edge, time_server)
        

    # receive lagging time from edge
    server_lagging = recv_float(socket_from_edge)

    # receive start time from edge
    time_start = recv_float(socket_from_edge)

    return server_lagging, time_start
    



def send_int(sock, integer):
    sock.send(integer.to_bytes(4, 'big', signed=True))

def recv_int(sock):
    received = recv_bytes(sock, 4)
    return int.from_bytes(received, 'big', signed=True)


def send_float(sock, floatnum):
    bytedata = struct.pack('d', floatnum)
    sock.send(bytedata)

def recv_float(sock):
    received = recv_bytes(sock, 8)
    return struct.unpack('d', received)[0]


def send_ndarray(sock, ndarray):
    # Send dtype
    dtype = ndarray.dtype
    if dtype == np.float64:
        dypte_code = DTYPE_FP64
    elif dtype == np.float32:
        dypte_code = DTYPE_FP32
    elif dtype == np.float16:
        dypte_code = DTYPE_FP16
    elif dtype == np.int32:
        dypte_code = DTYPE_INT32
    elif dtype == np.int16:
        dypte_code = DTYPE_INT16
    elif dtype == np.int8:
        dypte_code = DTYPE_INT8
    elif dtype == np.uint8:
        dypte_code = DTYPE_UINT8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    send_int(sock, dypte_code)
    
    # Send shape
    shape = ndarray.shape
    num_dims = len(shape)
    send_int(sock, num_dims)
    for dim in shape:
        send_int(sock, dim)

    # Send data
    bytedata = ndarray.tobytes()
    datasize = len(bytedata)
    send_int(sock, datasize)
    sock.send(bytedata)

def recv_ndarray(sock):
    # Receive dtype
    dtype_code = recv_int(sock)
    if dtype_code == DTYPE_FP64:
        dtype = np.float64
    elif dtype_code == DTYPE_FP32:
        dtype = np.float32
    elif dtype_code == DTYPE_FP16:
        dtype = np.float16
    elif dtype_code == DTYPE_INT32:
        dtype = np.int32
    elif dtype_code == DTYPE_INT16:
        dtype = np.int16
    elif dtype_code == DTYPE_INT8:
        dtype = np.int8
    elif dtype_code == DTYPE_UINT8:
        dtype = np.uint8
    else:
        raise ValueError(f"Unsupported dtype code: {dtype_code}")

    # Receive shape
    num_dims = recv_int(sock)
    shape = []
    for _ in range(num_dims):
        dim = recv_int(sock)
        shape.append(dim)

    # Receive data
    datasize = recv_int(sock)
    bytedata = recv_bytes(sock, datasize)
    ndarray = np.frombuffer(bytedata, dtype=dtype).reshape(shape)
    
    return ndarray