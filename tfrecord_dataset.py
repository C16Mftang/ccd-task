"""
Load .tfrecord movies data and metadata
"""

import os
import io
# import blosc
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import argparse

HOME_PATH = '/home/macleanlab/mufeng/NaturalMotionCNN/models/'
IMG_PATH = '/home/macleanlab/mufeng/tfrecord_data_processing/plots/'
PI = np.pi

def pool_fn(_a):
    np_bytes = blosc.decompress(_a)
    np_bytes_io = io.BytesIO(np_bytes)
    return np.load(np_bytes_io, allow_pickle=True)


def create_dataset(paths, max_seq_len=4800, encoding='png', pool=None):
    # for coherence change task: coherence_label and is_changes are variables of interest
    feature_description = {
        'frames': tf.io.VarLenFeature(tf.string),
        'change_label': tf.io.VarLenFeature(tf.int64),
        'coherence_label': tf.io.VarLenFeature(tf.int64),
        'direction_label': tf.io.VarLenFeature(tf.int64),
        'dominant_direction': tf.io.VarLenFeature(tf.int64),
        'trial_coherence': tf.io.VarLenFeature(tf.int64),
        'is_changes': tf.io.VarLenFeature(tf.int64),
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
            # frames: the encoded video
            _frames = tf.py_function(func=fn, inp=[_x['frames'].values[0]], Tout=tf.uint8)[:max_seq_len]

        _coh_label = _x['coherence_label'].values
        _change_label = _x['change_label'].values
        _direction_label = _x['direction_label'].values
        _dominant_direction = _x['dominant_direction'].values
        _trial_coherence = _x['trial_coherence'].values
        _is_changes = _x['is_changes'].values

        return _frames, dict(tf_op_layer_coherence=_coh_label, # 0 or 1, length 4800
                             tf_op_layer_change=_change_label, # 0 or 1, length 4800
                             tf_op_dir=_direction_label, # 1,2,3,4, length 4800
                             tf_dom_dir=_dominant_direction, # 1,2,3,4, length 10 (i.e. per trial/grey screen)
                             tf_trial_coh=_trial_coherence, # 0 or 1, length 10
                             tf_is_changes=_is_changes,
                            )

    data_set = data_set.map(_parse_example, num_parallel_calls=24)
    return data_set

def main():
    parser = argparse.ArgumentParser(description='Args for reading in tfrecord files')
    parser.add_argument('num_files', type=int, help='number of tfrecord files to read in')
    args = parser.parse_args()

    # for now, each tfrecord file corresponds to only one movie
    file_names = [os.path.expanduser(f'preprocessed/processed_data_{i+1}.tfrecord') for i in range(args.num_files)]
    data_set = create_dataset(file_names, 4800).batch(1)

    # ex[0] the frames (tensor), shape [1, 4800, 36, 64, 3]
    # ex[1] a dictionary, containing coh level, change label and direction
    k = 1
    movies = []
    trial_cohs = [] # original coherence level for each trial
    coh_labels = [] # coherence level at each frame
    is_changes = [] # whether coherence changes for each trial
    for ex in data_set: # iterate through the 30 movies
        trials = []
        print("Movie ", k)

        trial_coh = ex[1]['tf_trial_coh'].numpy()[0] # len = 10, coh vector of this movie
        trial_cohs.append(trial_coh)
        coh_label = ex[1]['tf_op_layer_coherence'].numpy()[0]
        coh_labels.append(coh_label)
        is_change = ex[1]['tf_is_changes'].numpy()[0]
        is_changes.append(is_change)

        # start and end frames of each trial movie (no gray screens)
        start_frames = np.arange(0, 80, 8)
        end_frames = np.arange(4, 84, 8)
        framerate = 60
        for i in range(len(trial_coh)): # 10
            trial = ex[0][:, start_frames[i]*framerate:end_frames[i]*framerate] # [1, 240, 36, 64, 3]
            trials.append(trial)

        movie = tf.concat(trials, axis=0) # [10, 240, 36, 64, 3]
        movies.append(movie)
        k+=1
    
    all_movies = tf.concat(movies, axis=0) # [300,240,36,64,3]
    all_trial_cohs = np.stack(trial_cohs)
    all_coh_labels = np.stack(coh_labels)
    all_is_changes = np.stack(is_changes)
    print(all_is_changes)
    print(all_trial_cohs)
    print(all_coh_labels.shape)

if __name__ == '__main__':
    main()



