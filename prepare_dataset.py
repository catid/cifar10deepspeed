import os

def create_file_list(base_dir, label_dict):
    file_list = []
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                if image_file.endswith('.png'):
                    file_path = os.path.abspath(os.path.join(label_dir, image_file))
                    file_list.append(f"{file_path} {label_dict[label]}")
    return file_list

def main():
    base_dir = 'cifar100'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # Extract labels and assign numbers
    labels = sorted(set(os.listdir(train_dir) + os.listdir(test_dir)))
    label_dict = {label: idx for idx, label in enumerate(labels)}

    # Write labels to labels.txt
    with open(f"{base_dir}/labels.txt", 'w') as f:
        for label, idx in label_dict.items():
            f.write(f"{label}, {idx}\n")

    # Write the number of labels to label_count.txt
    with open(f"{base_dir}/label_count.txt", 'w') as f:
        f.write(f"{len(labels)}\n")

    # Create file lists for training and validation
    training_file_list = create_file_list(train_dir, label_dict)
    validation_file_list = create_file_list(test_dir, label_dict)

    # Write file lists to respective text files
    with open(f"{base_dir}/training_file_list.txt", 'w') as f:
        for item in training_file_list:
            f.write(f"{item}\n")

    with open(f"{base_dir}/validation_file_list.txt", 'w') as f:
        for item in validation_file_list:
            f.write(f"{item}\n")

if __name__ == "__main__":
    main()
