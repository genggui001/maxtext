from tqdm.auto import tqdm
from collections import Counter
import json

# input_sentence_size = 35000000
input_sentence_size = 3600000

def data_loader():

    import tensorflow as tf
    import tensorflow_io as tfio
    import tensorflow_text as tftxt

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    SPECS = {
        "text": tf.TensorSpec(tf.TensorShape([]), tf.string, name="text"),
    }

    with tf.io.gfile.GFile("/home/genggui001/code/maxtext/assets/new_llama_add_chinese_other.model", 'rb') as model_fp:
        sp_model = model_fp.read()
        
    sp_tokenizer = tftxt.SentencepieceTokenizer(
        model=sp_model, add_bos=False, add_eos=False, reverse=False)
    
    import random

    random.seed(42)

    en_data_paths = sorted(
        tf.io.gfile.glob("/home/genggui001/gdrive/gg-nlp-lm-new/gg_en/**/*.jsonl.gz")
    )
    random.shuffle(en_data_paths)
    print(en_data_paths[:5])

    ds = tf.data.TextLineDataset(
        en_data_paths,
        compression_type="GZIP",
        buffer_size=8 * 1024 * 1024,
        num_parallel_reads=AUTOTUNE,
    )

    def decode(x):
        x = tfio.experimental.serialization.decode_json(x, specs=SPECS)
        x = sp_tokenizer.tokenize(x['text'])
        return  tf.boolean_mask(x, x > 59152)

    ds = ds.map(decode, num_parallel_calls=AUTOTUNE)
    ds = ds.take(input_sentence_size)

    for item in tqdm(ds.as_numpy_iterator()):
        yield item


counter = Counter()

for item in data_loader():
    counter.update(item.tolist())

with open("/home/genggui001/code/maxtext/assets/count.jsonl", "w", encoding="utf8") as f:
    json.dump(counter.most_common(), f, indent=4, ensure_ascii=False)


