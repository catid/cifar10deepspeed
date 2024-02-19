import socket
import numpy as np

def read_servers_from_hostfile(filepath):
    servers = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            server_address = line.split()[0]
            servers.append(f"{server_address}:5920")
    return servers

def distribute_task(command, server):
    server_address, server_port = server.split(':')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((server_address, int(server_port)))
        print(f"Sending task to {server_address}")
        sock.sendall(command.encode('utf-8'))
        # Wait for the task to complete
        response = sock.recv(1024)
        print(f"Response from server: {response.decode('utf-8')}")

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
    for command in commands:
        for server in servers:
            distribute_task(command, server)
            # In this example, tasks are sent sequentially to each server
            # Modify this loop for concurrent execution as needed

if __name__ == '__main__':
    main()
