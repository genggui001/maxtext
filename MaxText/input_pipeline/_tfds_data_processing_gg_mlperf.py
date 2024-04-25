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

"""Input pipeline for gpt3 c4 mlperf dataset."""

from typing import Optional

import functools

import numpy as np

import ml_collections
import tensorflow as tf
import tensorflow_io as tfio
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils

from sentencepiece import SentencePieceProcessor
import multihost_dataloading
import sequence_packing
import random
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


# data processing functions:
#   _shift_left_and_pad, rekey, reduce_concat_tokens and split_tokens_to_targets_length
# Adapted from:
#   https://github.com/google-research/text-to-text-transfer-transformer/blob/ba171b6f94eafcee60d0714fd6d60749b572d1f2/t5/data/preprocessors.py
# -----------------------------------------------------------------------------
def _shift_left_and_pad(tensor, pad_val):
    """Shift the input to the left with pad_val"""
    # Expand dims here so that the below code can work with 1-d tensors.
    v = tf.expand_dims(tensor, 0)
    # Make sure we keep tensor as ragged to allow for uneven concat.
    if isinstance(v, tf.Tensor):
        v = tf.RaggedTensor.from_tensor(v)

    # Append padding to the last item of every sequence.
    pad_shape = tf.concat([v.bounding_shape()[:-2], [1, 1]], axis=0)
    pad_tensor = tf.broadcast_to(pad_val, pad_shape)
    last_in_sequence = tf.concat([v[..., -1:, 1:], pad_tensor], axis=-1)
    # Concat back the newly modified final sequence item.
    v = tf.concat([v[..., :-1, :], last_in_sequence], axis=-2)
    # Un-expand outer dimension.
    v = v[0]
    return v


def loadjson_and_rekey(ds):
    """normalization with key mapping"""
    json_specs = {
        "text": tf.TensorSpec(tf.TensorShape([]), tf.string, name="text"),
    }
    key_map={"inputs": None, "targets": "text"}
    text_max_len = 10 * 1024 * 1024 # 10M

    def _loadjson_and_rekey(x, json_specs, key_map=None):
        """Replace the feature keys according to the mapping in `key_map`.
        For example, if the dataset returns examples of the format:
        {'foo': 'something', 'bar': 'something else', 'zoo': 'others'}
        and key_map = {'boo': 'foo', 'spar': 'bar', 'zoo': None} then this function will return
        examples with the format
        {'boo': 'something', 'spar': 'something else'}
        If a mapping is to None, then the key will be dropped.
        Args:
          x: an example to process.
          key_map: dictionary mapping new keys to original keys
        Returns:
          A preprocessed example with the format listed above.
        """
        x = tfio.experimental.serialization.decode_json(x, specs=json_specs)
        x["text"] = tf.strings.substr(x["text"], 0, text_max_len, unit='BYTE')

        x = {
            new_key: x[old_key] for new_key, old_key in key_map.items() if old_key
        }

        return x

    return ds.map(
        functools.partial(_loadjson_and_rekey, json_specs=json_specs, key_map=key_map), num_parallel_calls=AUTOTUNE
    )


def reduce_concat_tokens(
    dataset,
    feature_key="targets",
    batch_size=128,
):
    """Token-preprocessor to concatenate multiple unrelated documents.
    If we want to generate examples of exactly the right length,
    (to avoid wasting space on padding), then we use this function, followed by
    split_tokens.
    Args:
      dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
      feature_key: an string
      batch_size: an integer - how many documents to concatenate into one
    Returns:
      a dataset
    """
    dataset = dataset.map(
        lambda x: {feature_key: x[feature_key]}, num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes={feature_key: [-1]})

    def _my_fn(x):
        tokens = tf.reshape(x[feature_key], [-1])
        # strip padding
        tokens = tf.boolean_mask(tokens, tf.cast(tokens, tf.bool))
        return {feature_key: tokens}

    return dataset.map(_my_fn, num_parallel_calls=AUTOTUNE)


def split_tokens(
    dataset,
    max_tokens_per_segment=128,
    feature_key="targets",
):
    """Split examples into multiple examples each.
    The intended use case is to break up long examples for use in unsupervised
    transfer-learning.
    This function is generally preceded by select_random_chunk.
    Args:
      dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
      max_tokens_per_segment: an integer, the maximum number of tokens in each
        segment. Only the final segment may be shorter.
      feature_key: a string, the feature to split
    Returns:
      a dataset
    """

    def _split_tokens(x):
        """Split one token sequence into multiple multiple."""
        tokens = x[feature_key]
        n_tokens = tf.size(tokens)
        length = max_tokens_per_segment

        # Pad to a multiple of length, then use tf.reshape to split up the tokens
        # into num_segments segments each of the given length.
        num_segments = tf.cast(
            tf.math.ceil(tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32)),
            tf.int32,
        )
        padding = num_segments * length - tf.size(tokens)
        tokens = tf.pad(tokens, [[0, padding]])
        return tf.reshape(tokens, [-1, length])

    def _strip_padding(x):
        return {feature_key: tf.boolean_mask(x, tf.cast(x, tf.bool))}

    # Filter empty examples.
    dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))
    dataset = dataset.map(_split_tokens, num_parallel_calls=AUTOTUNE)
    dataset = dataset.unbatch()
    return dataset.map(_strip_padding, num_parallel_calls=AUTOTUNE)


def split_tokens_to_targets_length(dataset, sequence_length):
    return split_tokens(dataset, max_tokens_per_segment=sequence_length)


def _pad_to_batch_size(
    ds: tf.data.Dataset,
    batch_size: int,
    num_examples: Optional[int] = None,
) -> tf.data.Dataset:
    """Pad unevenly distributed eval data in each shard with new entries to multiples of batch size."""

    # local_num represents the total number of examples in eval dataset,
    if num_examples:
        local_num = num_examples
    else:

        def _get_num_examples(ds: tf.data.Dataset) -> int:
            # Iterate one-by-one instead of len(list(...)) to reduce peak memory.
            num_examples = 0
            for _ in ds:
                num_examples += 1

            return num_examples

        local_num = _get_num_examples(ds)
    local_num_batches = (local_num + batch_size - 1) // batch_size
    # Find the max number of batches required across all Jax processes.
    num_batches_all = multihost_utils.process_allgather(
        jnp.array([local_num_batches]), tiled=False
    )
    num_batches = np.max(num_batches_all)

    pad_num = num_batches * batch_size - local_num
    assert pad_num >= 0
    print(
        f"Eval data has {local_num} local entries, padding now with "
        f"{pad_num} extra entries to get {num_batches} batches."
    )

    # Repeat a random example to make the last batch full.
    def _add_pad(x):
        x["targets_segmentation"] *= 0
        return x

    pad_ds = ds.take(1).map(_add_pad).repeat(pad_num)
    return ds.concatenate(pad_ds)


def load_base_dataset(
    pattern,
    seed,
    data_index,
    data_num_shards,
):
    data_paths = sorted(tf.io.gfile.glob(pattern))

    # shard dataset now
    print((pattern, "all_file_count", len(data_paths)))
    # data_num_shards = jax.process_count()
    # data_index = jax.process_index()
    data_paths = [
        d
        for i, d in enumerate(data_paths)
        if i % data_num_shards == data_index
    ]
    random.seed(seed + data_index)
    random.shuffle(data_paths)

    print((pattern, "share_file_count", data_num_shards, data_index, len(data_paths), data_paths[:2]))

    ds = tf.data.TextLineDataset(
        data_paths,
        compression_type="GZIP",
        buffer_size=8 * 1024 * 1024,
        num_parallel_reads=2,
    )

    return ds

def get_datasets(
    config: ml_collections.ConfigDict,
    dataloading_host_index,
    dataloading_host_count,
):
    """Load and return dataset of batched examples for use during training."""
    en_ds = load_base_dataset(
        pattern=os.path.join(config.dataset_path, "gg_en/**/*.jsonl.gz"),
        seed=config.data_shuffle_seed,
        data_index=dataloading_host_index,
        data_num_shards=dataloading_host_count,
    )

    zh_ds = load_base_dataset(
        pattern=os.path.join(config.dataset_path, "gg_zh/**/*.jsonl.gz"),
        seed=config.data_shuffle_seed,
        data_index=dataloading_host_index,
        data_num_shards=dataloading_host_count,
    )

    other_ds = load_base_dataset(
        pattern=os.path.join(config.dataset_path, "uonlp_culturax_shuffle/**/*.jsonl.gz"),
        seed=config.data_shuffle_seed,
        data_index=dataloading_host_index,
        data_num_shards=dataloading_host_count,
    )

    code_ds = load_base_dataset(
        pattern=os.path.join(config.dataset_path, "the-stack-dedup/**/*.jsonl.gz"),
        seed=config.data_shuffle_seed,
        data_index=dataloading_host_index,
        data_num_shards=dataloading_host_count,
    )

    train_ds = tf.data.Dataset.sample_from_datasets(
        datasets = [
            en_ds.repeat(),
            zh_ds.repeat(),
            other_ds.repeat(),
            code_ds.repeat(),
        ], 
        weights=[
            0.5,
            0.2,
            0.2,
            0.1,
        ],
        seed=config.data_shuffle_seed,
    )

    eval_ds = train_ds.take(int(config.eval_dataset_size))

    # shard the dataset as soon as it is loaded
    # not use
    # train_ds = train_ds.shard(num_shards=jax.process_count(), index=jax.process_index())
    train_ds = loadjson_and_rekey(
        train_ds, 
    )

    # nor use
    # eval_ds = eval_ds.shard(num_shards=jax.process_count(), index=jax.process_index())
    # note validation_tokenized_5662seqs split is pre tokenized, reduce_concated and split to target_length
    #   mainly to avoid eval sequences change depending on the number of hosts
    eval_ds = loadjson_and_rekey(
        eval_ds, 
    )

    return train_ds, eval_ds


def preprocess_dataset(
    config: ml_collections.ConfigDict,
    dataloading_host_index,
    dataloading_host_count,
    global_mesh,
    train_ds,
    eval_ds,
    vocab_path,
    add_bos=True,
    add_eos=True,
    data_shuffle_seed: int = 0,
    shuffle_buffer_size: int = 4096,
):
    # Set global batch size.
    global_batch_size_to_load = config.global_batch_size_to_load

    if config.eval_per_device_batch_size > 0:
        eval_batch_size = config.eval_per_device_batch_size * global_mesh.size
    else:
        eval_batch_size = global_batch_size_to_load
    
    # Set tokenize_processor
    tokenize_processor = SentencePieceProcessor()
    tokenize_processor.Load(vocab_path)

    def tokenize_fn(t):
        token_ids = tokenize_processor.EncodeAsIds(t)
        if add_bos:
            token_ids = [tokenize_processor.bos_id()] + token_ids
        if add_eos:
            token_ids = token_ids + [tokenize_processor.eos_id()]
        return np.asarray(token_ids, dtype=np.int32)

    # train
    train_ds = train_ds.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)
    train_ds = train_ds.map(
        lambda x: {
            'targets': tf.numpy_function(
                func=tokenize_fn, 
                inp=[x['targets']], 
                Tout=tf.int32, 
                stateful=False,
            )
        },
        num_parallel_calls=AUTOTUNE,
    )
    train_ds = reduce_concat_tokens(train_ds, feature_key="targets", batch_size=512)
    train_ds = split_tokens_to_targets_length(train_ds,  config.max_target_length+1)

    def train_format_fn(x):
        tokens = x["targets"]

        x["inputs"] = tokens[:-1]
        x["targets"] = tokens[1:]
        
        x["inputs_segmentation"] = tf.ones_like(x["inputs"])
        x["targets_segmentation"] = x["inputs_segmentation"]

        x["inputs_position"] = tf.range(tf.size(tokens)-1, dtype=tf.int32)
        x["targets_position"] = x["inputs_position"]

        return x

    train_ds = train_ds.map(train_format_fn, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.padded_batch(
        global_batch_size_to_load // jax.process_count(), 
        padded_shapes={
            "inputs": config.max_target_length,
            "targets": config.max_target_length,
            "inputs_segmentation": config.max_target_length,
            "targets_segmentation": config.max_target_length,
            "inputs_position": config.max_target_length,
            "targets_position": config.max_target_length,
        },
        padding_values={
            "inputs": 0,
            "targets": 0,
            "inputs_segmentation": 0,
            "targets_segmentation": 0,
            "inputs_position": 0,
            "targets_position": 0,
        },
        drop_remainder=True
    )

    train_ds = train_ds.prefetch(32)

    # eval_ds
    eval_ds = eval_ds.map(
        lambda x: {
            'targets': tf.numpy_function(
                func=tokenize_fn, 
                inp=[x['targets']], 
                Tout=tf.int32, 
                stateful=False,
            )
        },
        num_parallel_calls=AUTOTUNE,
    )
    # note eval_ds is pre tokenized, reduce_concated and split to target_length
    #   mainly to avoid eval sequences change depending on the number of hosts
    eval_ds = sequence_packing.pack_dataset(eval_ds, config.max_target_length+1)

    def eval_format_fn(x):
        x["inputs"] = x["targets"][:-1]
        x["inputs_position"] = x["targets_position"][:-1]
        x["inputs_segmentation"] = x["targets_segmentation"][:-1]
        
        x["targets"] = x["targets"][1:]
        x["targets_position"] = x["targets_position"][1:]
        x["targets_segmentation"] = x["targets_segmentation"][1:]
        return x

    eval_ds = eval_ds.map(eval_format_fn, num_parallel_calls=AUTOTUNE)

    # ensure array split in an equal division for each device
    # pad zeros up to the same batch_size among all processes
    eval_ds = _pad_to_batch_size(eval_ds, eval_batch_size // jax.process_count())

    eval_ds = eval_ds.batch(
        eval_batch_size // jax.process_count(), drop_remainder=False
    )
    # We are running eval over exactly one epoch.
    # We explicitly cache the entire epoch (in memory) to ensure that it is the
    # same across different iterations.
    eval_ds = eval_ds.cache()

    train_multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(
        dataloader=train_ds, 
        global_mesh=global_mesh,
        length=None,
        dataloder_save_directory=os.path.join(config.dataloader_checkpoint_dir, f"dataloader-{dataloading_host_index}-{dataloading_host_count}"),
        dataloder_max_to_keep=config.checkpoint_max_to_keep if config.checkpoint_max_to_keep > 0 else None,
    )
    eval_multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(
        dataloader=eval_ds, 
        global_mesh=global_mesh,
        length=None,
        dataloder_save_directory=None,
        dataloder_max_to_keep=None,
    )

    # Return multi-host jax.Array prep iterator
    return train_multihost_gen, eval_multihost_gen
