# This uses Nvidia DALI to batch load the images/labels

import torch
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from functools import partial
import numpy as np

@pipeline_def(batch_size=64, num_threads = 8, exec_async=False, exec_pipelined=False)
def png_pipeline(data_dir=None, file_list=None, file_index_map=None,
                 mode="training", crop_w=32, crop_h=32, shard_id=0, num_shards=1):

    file_names, labels = fn.readers.file(
        file_root=data_dir,
        file_list=file_list,
        random_shuffle=(mode == 'training'),
        name="Reader",
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=True)

    decoded_images = fn.decoders.image(
        file_names,
        device="mixed",
        output_type=types.RGB)

    if mode == 'training':
        # Data Augmentations for Training:

        # Pick a random crop from each input image
        # Mirror the images horizontally 40% of the time
        images = fn.crop_mirror_normalize(
            decoded_images,
            device="gpu",
            dtype=types.UINT8,
            crop=(crop_h, crop_w),
            crop_pos_x=0.5,
            crop_pos_y=0.5,
            output_layout=types.NHWC,
            #mean=[0.49139968, 0.48215827, 0.44653124],
            #std=[0.24703233, 0.24348505, 0.26158768],
            mirror=fn.random.coin_flip(probability=0.3))

        images = fn.random_resized_crop(images, size=(crop_h, crop_w), random_area=[0.25, 1.0], random_aspect_ratio=[0.9, 1.1])

        # Convert to NCHW
        normalized_full_images = fn.transpose(images, device="gpu", perm=[2, 0, 1])
    else:
        normalized_full_images = fn.crop_mirror_normalize(
            decoded_images,
            device="gpu",
            dtype=types.UINT8,
            crop=(crop_h, crop_w),
            crop_pos_x=0.5,
            crop_pos_y=0.5,
            output_layout=types.NCHW,
            #mean=[0.49139968, 0.48215827, 0.44653124],
            #std=[0.24703233, 0.24348505, 0.26158768],
            mirror=0)

    file_path = fn.get_property(file_names, key="source_info")

    def pad_file_path(file_path, file_index_map):
        s = file_path.tostring().decode()
        index = file_index_map[s]
        return np.array([index])

    pad_file_path_curried = partial(pad_file_path, file_index_map=file_index_map)

    file_path = fn.python_function(file_path, function=pad_file_path_curried, num_outputs=1)

    return labels, normalized_full_images, file_path

class CifarDataLoader:
    def __init__(self, batch_size, device_id, num_threads, seed, data_dir=None, file_list=None, mode='training',
                    crop_w=32, crop_h=32, shard_id=1, num_shards=1):

        file_index_map = {}
        with open(file_list, "r") as f:
            for idx, line in enumerate(f):
                # Split each line by whitespace to extract the filename
                filename = line.split()[0]
                # Assign an index to the filename if it's not already in the map
                if filename not in file_index_map:
                    file_index_map[filename] = idx

        self.pipeline = png_pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
            shard_id=shard_id,
            num_shards=num_shards,
            data_dir=data_dir,
            file_list=file_list,
            file_index_map=file_index_map,
            mode=mode,
            crop_w=crop_w,
            crop_h=crop_h)
        self.pipeline.build()
        self.loader = DALIGenericIterator(
            [self.pipeline],
            output_map=["labels", "images", "file_path"],
            reader_name="Reader",
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL)

    def __iter__(self):
        return self.loader.__iter__()

    def __len__(self):
        return len(self.loader)
