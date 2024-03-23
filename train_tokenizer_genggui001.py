import io
import glob
import sentencepiece as spm
from tqdm.auto import tqdm

input_sentence_size = 40000000

def data_loader():

    import tensorflow as tf
    import tensorflow_io as tfio

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    SPECS = {
        "text": tf.TensorSpec(tf.TensorShape([]), tf.string, name="text"),
    }

    # input_sentence_size = 50000000

    import random

    random.seed(42)

    # zh_data_paths = sorted(
    #     glob.glob("/home/genggui001/gdrive/gg-nlp-lm-new/gg_en/**/*.jsonl.gz", recursive=True)
    # )
    # random.shuffle(zh_data_paths)

    zh_data_paths = [
        "/home/genggui001/gdrive/gg-nlp-lm-new/gg_en/SlimPajama-627B/chunk-00000.jsonl.gz",
        "/home/genggui001/gdrive/gg-nlp-lm-new/gg_en/WebText-en/chunk-00000.jsonl.gz",
        "/home/genggui001/gdrive/gg-nlp-lm-new/gg_en/falcon-refinedweb/chunk-00000.jsonl.gz",
        "/home/genggui001/gdrive/gg-nlp-lm-new/gg_en/the_pile_deduplicated/chunk-00000.jsonl.gz",

        "/home/genggui001/gdrive/gg-nlp-lm-new/gg_zh/CCI-Data/chunk-00000.jsonl.gz",
        "/home/genggui001/gdrive/gg-nlp-lm-new/gg_zh/SkyPile-150B/chunk-00000.jsonl.gz",
        "/home/genggui001/gdrive/gg-nlp-lm-new/gg_zh/TeleChat-PTD/chunk-00000.jsonl.gz",
        "/home/genggui001/gdrive/gg-nlp-lm-new/gg_zh/WebText-cn/chunk-00000.jsonl.gz",
        "/home/genggui001/gdrive/gg-nlp-lm-new/gg_zh/WuDaoCorpus2.0/chunk-00000.jsonl.gz",
        "/home/genggui001/gdrive/gg-nlp-lm-new/gg_zh/wangan/chunk-00000.jsonl.gz",
        "/home/genggui001/gdrive/gg-nlp-lm-new/gg_zh/yayi2_pretrain_data/chunk-00000.jsonl.gz",

        "/home/genggui001/gdrive/gg-nlp-lm-new/the-stack-dedup/shuffle_data/schunk-00000.jsonl.gz",

        "/home/genggui001/gdrive/gg-nlp-lm-new/gg_others_shuffle/s/schunk-00000.jsonl.gz",
    ]

    ds = tf.data.TextLineDataset(
        zh_data_paths,
        compression_type="GZIP",
        buffer_size=8 * 1024 * 1024,
        num_parallel_reads=len(zh_data_paths),
    )
    text_max_len = 50
    def decode(x):
        x = tfio.experimental.serialization.decode_json(x, specs=SPECS)

        x["text"] = tf.strings.substr(x["text"], 0, text_max_len, unit='BYTE')
        return x['text']

    ds = ds.map(decode, num_parallel_calls=AUTOTUNE)
    ds = ds.take(int(input_sentence_size * 1.2))

    for item in (ds.as_numpy_iterator()):
        yield item


max_text_len = -1
for text in data_loader():
    text_len = len(text)
    if text_len > max_text_len:
        max_text_len = text_len
        print(max_text_len)

