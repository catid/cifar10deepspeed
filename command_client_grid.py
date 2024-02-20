import socket
import numpy as np
from threading import Thread
import time

def read_servers_from_hostfile(filepath):
    servers = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            server_address = line.split()[0]
            servers.append(f"{server_address}:5920")
    return servers

def send_task_to_server(command, server, completion_callback):
    server_address, server_port = server.split(':')
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((server_address, int(server_port)))
            print(f"Sending task to {server_address}")
            sock.sendall(command.encode('utf-8'))
            # Wait for the task to complete
            response = sock.recv(1024)
            print(f"Response from server {server_address}: {response.decode('utf-8')}")
    except Exception as e:
        print(f"Failed to send task to {server_address}: {e}")
    finally:
        completion_callback(server)

def task_distributor(commands, servers):
    available_servers = servers.copy()
    
    def mark_server_available(server):
        available_servers.append(server)
        distribute_next_task()
    
    def distribute_next_task():
        while available_servers and commands:
            server = available_servers.pop(0)
            command = commands.pop(0)
            Thread(target=send_task_to_server, args=(command, server, mark_server_available)).start()
    
    distribute_next_task()

def generate_commands(lr_start, lr_end, lr_steps, wd_start, wd_end, wd_steps):
    learning_rates = np.linspace(lr_start, lr_end, lr_steps)
    weight_decays = np.linspace(wd_start, wd_end, wd_steps)
    for lr in learning_rates:
        for wd in weight_decays:
            yield f"./launch_local_train.sh --wandb --lr {lr} --weight-decay {wd} --reset --name \"lr{lr}wd{wd}\""

def main():
    servers = read_servers_from_hostfile("hostfile")
    if not servers:
        print("No servers found in hostfile.")
        return

    lr_start, lr_end, lr_steps = 0.0001, 0.01, 8
    wd_start, wd_end, wd_steps = 0.0001, 0.01, 8

    commands = list(generate_commands(lr_start, lr_end, lr_steps, wd_start, wd_end, wd_steps))
    task_distributor(commands, servers)

if __name__ == '__main__':
    main()
