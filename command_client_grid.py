import socket
import numpy as np

def read_servers_from_hostfile(filepath):
    servers = []
    with open(filepath, 'r') as file:
        for line in file:
            # Ignore comments and empty lines
            if line.startswith('#') or not line.strip():
                continue
            server_address = line.split()[0]  # Assuming the first part is the server address
            servers.append(f"{server_address}:9200")  # Append default port
    return servers

def distribute_task(command, servers):
    for server in servers:
        server_address, server_port = server.split(':')
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((server_address, int(server_port)))
                sock.sendall(command.encode('utf-8'))
                print(f"Task sent to {server_address}")
                break  # Break after successful send
        except ConnectionError as e:
            print(f"Could not connect to {server_address}: {e}")

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

    # Define the sweep parameters
    lr_start, lr_end, lr_steps = 0.0001, 0.01, 8
    wd_start, wd_end, wd_steps = 0.0001, 0.01, 8

    commands = list(generate_commands(lr_start, lr_end, lr_steps, wd_start, wd_end, wd_steps))
    for i, command in enumerate(commands):
        server = servers[i % len(servers)]  # Round-robin distribution
        distribute_task(command, [server])  # Send task to one server at a time

if __name__ == '__main__':
    main()
