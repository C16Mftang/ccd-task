import os
import io
import blosc
import numpy as np
import tensorflow as tf


def pool_fn(_a):
    np_bytes = blosc.decompress(_a)
    np_bytes_io = io.BytesIO(np_bytes)
    return np.load(np_bytes_io, allow_pickle=True)


def create_dataset(paths, max_seq_len=3500, encoding='png', pool=None):
    # again, you will change the features to reflect the variables in your own metadata
    # you may also change the max_seq_len (which is the maximum duration for each trial in ms)
    feature_description = {
        'frames': tf.io.VarLenFeature(tf.string),
        'change_label': tf.io.VarLenFeature(tf.int64),
        'coherence_label': tf.io.VarLenFeature(tf.int64),
        'direction_label': tf.io.VarLenFeature(tf.int64)
    }
    data_set = tf.data.TFRecordDataset(paths)

    def _parse_example(_x):
        _x = tf.io.parse_single_example(_x, feature_description)

        if encoding == 'png':
            _frames = tf.map_fn(lambda a: tf.io.decode_png(a), _x['frames'].values, dtype=tf.uint8)[:max_seq_len]
        elif encoding == 'blosc':
            def fn(_a):
                if pool is None:
                    return pool_fn(_a.numpy())
                else:
                    return pool.apply(pool_fn, (_a.numpy(),))

            _frames = tf.py_function(func=fn, inp=[_x['frames'].values[0]], Tout=tf.uint8)[:max_seq_len]
            # _frames = tf.zeros((max_seq_len, 125, 200, 1), tf.uint8)

        _m1 = tf.cast(tf.random.uniform(minval=0, maxval=1, shape=()) > .5, tf.int32) * 2 - 1
        _m2 = tf.cast(tf.random.uniform(minval=0, maxval=1, shape=()) > .5, tf.int32) * 2 - 1
        _frames = _frames[:, ::_m1, ::_m2]
        _seq_len = tf.shape(_frames)[0]
        _p1 = [0, max_seq_len - _seq_len]
        _p = [_p1, [0, 0], [0, 0], [0, 0]]

        _frames = tf.pad(_frames, _p)

        _label = tf.pad(_x['coherence_label'].values[:max_seq_len], [_p1])
        _change_label = tf.pad(_x['change_label'].values[:max_seq_len], [_p1])
        _change_label += tf.pad(_change_label[:-23], [[23, 0]])
        #_label += tf.pad(_label[:-23*2], [[23*2, 0]])
        # _x = {'frames': _frames, 'label': _label}
        return _frames, dict(tf_op_layer_coherence=_label, tf_op_layer_change=_change_label)

    data_set = data_set.map(_parse_example, num_parallel_calls=24)
    return data_set


def main():
    file_names = [os.path.expanduser('~/data/processed_data_1.tfrecord')]

    data_set = create_dataset(file_names, 1000).batch(16)

    for ex in data_set:
        # print({k: v.shape for k, v in ex.items()})
        print([a.shape for a in ex])


if __name__ == '__main__':
    main()
