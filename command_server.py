# This script can be used from `command_client_grid.py` to distribute grid search tasks.

import socket
import subprocess
from threading import Thread

def handle_client_connection(client_socket):
    try:
        while True:
            command = client_socket.recv(1024).decode('utf-8')
            if not command:
                break  # Client closed connection
            print(f"Executing: {command}")
            process = subprocess.Popen(command, shell=True)
            process.wait()
            client_socket.sendall(b"Task Completed")  # Send completion message back to client
    finally:
        client_socket.close()

def main():
    host = '0.0.0.0'
    port = 5920

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"[*] Listening on {host}:{port}")

    try:
        while True:
            client_sock, address = server.accept()
            print(f"[*] Accepted connection from {address[0]}:{address[1]}")
            client_handler = Thread(target=handle_client_connection, args=(client_sock,))
            client_handler.start()
    finally:
        server.close()

if __name__ == '__main__':
    main()
