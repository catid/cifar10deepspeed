# This uses Nvidia DALI to batch load the images/labels

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.pytorch as dali_torch
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

@pipeline_def(batch_size=64, num_threads = 8, exec_async=False, exec_pipelined=False)
def png_pipeline(data_dir=None, file_list=None, mode="training", crop_w=32, crop_h=32,
                shard_id=0, num_shards=1):

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
            #crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
            #crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
            output_layout=types.NHWC,
            mirror=fn.random.coin_flip(probability=0.3))

        # 50% of the time, apply a random rotation of 90, 180, or 270 degrees
        angle = fn.random.coin_flip(probability=0.3) * fn.random.uniform(range=(1, 4), dtype=dali.types.INT32) * 90.0
        images = fn.rotate(images, device="gpu", angle=angle)

        images = fn.random_resized_crop(images, size=(crop_h, crop_w), random_area=[0.25, 1.0], random_aspect_ratio=[1.0, 1.0])

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
            mirror=0)

    return labels, normalized_full_images

class CustomDALIIterator(dali_torch.DALIGenericIterator):
    def __init__(self, pipelines, *args, **kwargs):
        super(CustomDALIIterator, self).__init__(pipelines, ["labels", "full"], *args, **kwargs)

    def __next__(self):
        out = super().__next__()
        out = out[0]

        # Extract the downsampled and upsampled images from the output
        labels = out["labels"]
        full = out["full"]

        return labels, full

class CifarDataLoader:
    def __init__(self, batch_size, device_id, num_threads, seed, data_dir=None, file_list=None, mode='training',
                    crop_w=32, crop_h=32, shard_id=1, num_shards=1):
        self.pipeline = png_pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
            shard_id=shard_id,
            num_shards=num_shards,
            data_dir=data_dir,
            file_list=file_list,
            mode=mode,
            crop_w=crop_w,
            crop_h=crop_h)
        self.pipeline.build()
        self.loader = CustomDALIIterator(
            [self.pipeline],
            reader_name="Reader",
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.PARTIAL)

    def __iter__(self):
        return self.loader.__iter__()

    def __len__(self):
        return len(self.loader)
