"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""Input pipeline using Grain."""

import os
import glob
from typing import Optional, SupportsIndex
from tqdm.auto import tqdm

import ml_collections
import jax
import grain.python as grain

# import tokenizer
from input_pipeline import _grain_operations
from input_pipeline import _grain_tokenizer

import multihost_dataloading


class SArrayRecordDataSource:
    def __init__(
        self,
        data_paths,
        d_group,
        group_size,
    ):
        assert sum(d_group) == group_size
        assert len(data_paths) == len(d_group)

        for data_path in data_paths:
            print(("read_data", data_path[:2], len(data_path)))

        self._data_paths = data_paths
        self._d_group = d_group
        self._group_size = group_size

        self._datasets = [
            grain.ArrayRecordDataSource(data_path) for data_path in tqdm(data_paths)
        ]

        self._lengths = [len(dset) for dset in self._datasets]

        # 最大的一个迭代完成 * 10 (铁定完不成)
        self._len = max(
            [int(l / g * group_size) for l, g in zip(self._lengths, d_group)]
        ) * 10

        self._offsets = []
        for d_i, g_size in enumerate(d_group):
            self._offsets += [(d_i, g_s_i, g_size) for g_s_i in range(g_size)]

        import numpy as np

        rng = np.random.default_rng(42)
        rng.shuffle(self._offsets)

        assert len(self._offsets) == group_size

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, record_key: SupportsIndex):
        record_key = record_key.__index__()
        assert record_key >= 0 and record_key < self._len

        g_i, offset = divmod(record_key, self._group_size)
        d_i, g_s_i, g_size = self._offsets[offset]
        d_offset = (g_i * g_size + g_s_i) % self._lengths[d_i]

        return self._datasets[d_i][d_offset]

    def __str__(self):
        return f"SArrayRecordDataSource(data_paths={self._data_paths}, d_group={self._d_group}, group_size={self._group_size}, "


def get_datasets(config: ml_collections.ConfigDict):
    """Load dataset from array_record files for using with grain"""
    train_ds = SArrayRecordDataSource(
        data_paths = [
            sorted(glob.glob(os.path.join(config.dataset_path, "gg_en_array_record/*/*.jsonl.array_record"))),
            sorted(glob.glob(os.path.join(config.dataset_path, "gg_zh_array_record/*/*.jsonl.array_record"))),
            sorted(glob.glob(os.path.join(config.dataset_path, "uonlp_culturax_shuffle/*/*.jsonl.array_record"))),
            sorted(glob.glob(os.path.join(config.dataset_path, "the-stack-dedup_data_record/*/*.jsonl.array_record"))),
        ],
        d_group=[
            9, 
            4, 
            5, 
            2
        ],
        group_size=20,
    )

    return train_ds, train_ds


def preprocess_dataset(
    config: ml_collections.ConfigDict,
    global_mesh,
    train_ds,
    eval_ds,
    vocab_path: Optional[str] = None,
    data_shuffle_seed=0,
    add_bos=True,
    add_eos=True,
):
    assert config.eval_step > 0
    """Use grain to pre-process the dataset and return iterators"""
    # Set global batch size.
    global_batch_size_to_load = config.global_batch_size_to_load

    if config.eval_per_device_batch_size > 0:
        eval_batch_size = config.eval_per_device_batch_size * global_mesh.size
    else:
        eval_batch_size = global_batch_size_to_load

    train_iter = preprocessing_pipeline(
        train_ds,
        vocab_path,
        add_bos,
        add_eos,
        config.grain_worker_count,
        global_batch_size_to_load,
        global_mesh,
        shuffle=config.enable_data_shuffling,
        pack_examples=True,
        max_length=config.max_target_length,
        data_shuffle_seed=data_shuffle_seed,
        data_iter_count=None,
    )

    eval_iter = preprocessing_pipeline(
        eval_ds,
        vocab_path,
        add_bos,
        add_eos,
        config.grain_worker_count,
        eval_batch_size,
        global_mesh,
        shuffle=config.enable_data_shuffling,
        pack_examples=True,
        max_length=config.max_target_length,
        data_shuffle_seed=data_shuffle_seed+1,
        data_iter_count=config.eval_step,
    )

    predict_iter = preprocessing_pipeline(
        eval_ds,
        vocab_path,
        add_bos,
        add_eos,
        config.grain_worker_count,
        eval_batch_size,
        global_mesh,
        shuffle=config.enable_data_shuffling,
        pack_examples=True,
        max_length=config.max_target_length,
        data_shuffle_seed=data_shuffle_seed+1,
        data_iter_count=config.eval_step,
    )

    return train_iter, eval_iter, predict_iter


def preprocessing_pipeline(
    dataset,
    vocab_path,
    add_bos: bool,
    add_eos: bool,
    grain_worker_count: int,
    batch_size: int,
    global_mesh,
    shuffle: bool,
    pack_examples: bool = True,
    max_length: int = 512,
    drop_remainder: bool = True,
    data_shuffle_seed: int = 0,
    data_iter_count: Optional[int] = None,
):
    """Apply grain operations to preprocess the given dataset."""
    assert (
        batch_size % global_mesh.size == 0
    ), "Batch size should be divisible number of global devices."

    operations = []
    operations.append(_grain_operations.GGParseFeatures())
    operations.append(
        _grain_tokenizer.TokenizeAndTrim(
            ["inputs"], max_length+1, vocab_path, add_bos, add_eos
        )
    )
    # Shift inputs for teacher-forced training
    operations.append(_grain_operations.GGShiftData())

    # Pack and Batch examples.
    if pack_examples:
        operations.append(
            grain.experimental.PackAndBatchOperation(
                batch_size=batch_size // jax.process_count(),
                length_struct={"inputs": max_length, "targets": max_length},
            )
        )
        operations.append(_grain_operations.ReformatPacking())
    else:
        operations.append(_grain_operations.PadToMaxLength(max_length))
        operations.append(
            grain.Batch(
                batch_size=batch_size // jax.process_count(),
                drop_remainder=drop_remainder,
            )
        )

    index_sampler = grain.IndexSampler(
        num_records=len(dataset),
        num_epochs=1,
        shard_options=grain.ShardOptions(
            shard_index=jax.process_index(),
            shard_count=jax.process_count(),
            drop_remainder=True,
        ),
        shuffle=shuffle,
        seed=data_shuffle_seed,
    )

    dataloader = grain.DataLoader(
        data_source=dataset,
        operations=operations,
        sampler=index_sampler,
        worker_count=grain_worker_count,
    )

    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(
        dataloader=dataloader, 
        global_mesh=global_mesh,
        length=data_iter_count,
    )

    # Return multi-host jax.Array prep iterator
    return multihost_gen
