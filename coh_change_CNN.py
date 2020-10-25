import os
import io
import numpy as np
import tensorflow as tf

def pool_fn(_a):
    np_bytes = blosc.decompress(_a)
    np_bytes_io = io.BytesIO(np_bytes)
    return np.load(np_bytes_io, allow_pickle=True)

# max_seq_len: expected length of the whole movie, containing both trials and gray screens, in ms
def create_dataset(paths, max_seq_len=3500, encoding='png', pool=None):
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

        # _m1 = tf.cast(tf.random.uniform(minval=0, maxval=1, shape=()) > .5, tf.int32) * 2 - 1
        # _m2 = tf.cast(tf.random.uniform(minval=0, maxval=1, shape=()) > .5, tf.int32) * 2 - 1
        # _frames = _frames[:, ::_m1, ::_m2]
        _seq_len = tf.shape(_frames)[0]
        _p1 = [0, max_seq_len - _seq_len]
        _p = [_p1, [0, 0], [0, 0], [0, 0]]
        # padd the frames with frames full of 0's until max_seq_len
        _frames = tf.pad(_frames, _p)
        # also padd all the feature vectors, otherwise will raise an error: number of elements doesn't match
        _label = tf.pad(_x['coherence_label'].values[:max_seq_len], [_p1])
        _change_label = tf.pad(_x['change_label'].values[:max_seq_len], [_p1])
        _change_label += tf.pad(_change_label[:-23], [[23, 0]])
        _direction_label = tf.pad(_x['direction_label'].values[:max_seq_len]. [_p1])
        return _frames, dict(tf_op_layer_coherence=_label, tf_op_layer_change=_change_label, tf_op_dir=_direction_label)

    data_set = data_set.map(_parse_example, num_parallel_calls=24)
    return data_set

def test_one_file():
    file_names = [os.path.expanduser('preprocessed/processed_data_1.tfrecord')]
    # tf dataset's .batch method combines consecutive elements into batches
    # for now each tfrecord file contains only one movie for testing their compatibility with CNN
    data_set = create_dataset(file_names, 3500).batch(1)

    for ex in data_set:
        print([type(a) for a in ex])
        print(tf.shape(ex[0]))
        dir_arr = ex[1]['tf_op_dir'].numpy()
        with np.printoptions(threshold=np.inf):
            print(dir_arr.shape)


if __name__ == '__main__':
    main()
