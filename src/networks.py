# deterministic receiver

def receive_socket(sock, bytes_to_receive):
    received = b''
    while len(received) < bytes_to_receive:
        received += sock.recv(bytes_to_receive - len(received))
    return received