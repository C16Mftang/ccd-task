import os
import io
import blosc
import numpy as np
import tensorflow as tf
import keras

HOME_PATH = '/home/macleanlab/mufeng/NaturalMotionCNN/models/'
PI = np.pi

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
        'direction_label': tf.io.VarLenFeature(tf.int64),
        'dominant_direction': tf.io.VarLenFeature(tf.int64),
        'trial_coherence': tf.io.VarLenFeature(tf.int64)
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

        # This whole padding section is not needed, as we have movies and trials of fixed size
        """
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
        _direction_label = tf.pad(_x['direction_label'].values[:max_seq_len], [_p1])
        _dominant_direction = _x['dominant_direction'].values[:max_seq_len]
        _trial_coherence = _x['trial_coherence'].values[:max_seq_len]
        """

        _label = _x['coherence_label'].values
        _change_label = _x['change_label'].values
        _direction_label = _x['direction_label'].values
        _dominant_direction = _x['dominant_direction'].values
        _trial_coherence = _x['trial_coherence'].values

        return _frames, dict(tf_op_layer_coherence=_label, # 0 or 1, length 4800
                             tf_op_layer_change=_change_label, # 0 or 1, length 4800
                             tf_op_dir=_direction_label, # 1,2,3,4, length 4800
                             tf_dom_dir=_dominant_direction, # 1,2,3,4, length 10 (i.e. per trial/grey screen)
                             tf_trial_coh=_trial_coherence, # 0 or 1, length 10
                             )

    data_set = data_set.map(_parse_example, num_parallel_calls=24)
    return data_set


# angs an (n,1) array
def angs_to_categories(angs):
    categories = np.zeros_like(angs)
    for i in range(angs.shape[0]):
        ang = angs[i,0]
        if ang >= -PI/4 and ang < PI/4:
            categories[i,0] = 1
        elif ang >= PI/4 and ang < 3*PI/4:
            categories[i,0] = 2
        elif (ang >= 3*PI/4 and ang <= PI) or (ang >= -PI and ang < -3*PI/4):
            categories[i,0] = 3
        elif ang >= -3*PI/4 and ang < -PI/4:
            categories[i,0] = 4
    return categories


def main():
    file_names = [os.path.expanduser(f'preprocessed/processed_data_{i+1}.tfrecord') for i in range(30)]
    data_set = create_dataset(file_names, 4800).batch(1)

    # ex[0] the frames (tensor), shape [1, 4800, 36, 64, 3]
    # ex[1] a dictionary, containing coh level, change label and direction
    k = 1
    """
    we want each movie to be represented by a list of trials, each trial to be represented by a 
    4d tensor of shape (batch,240,36,64,3); for each trial in this list, pass it through the CNN.
    """
    movies = []
    true_dirs = []
    trial_cohs = []
    for ex in data_set:
        trials = []
        print("Movie ", k)
        # the direction vector, of length max_seq_len, fixed for each trial/gray screen in the movie
        dirs = ex[1]['tf_op_dir'].numpy()[0]
        dom_dir = ex[1]['tf_dom_dir'].numpy()[0] # len = 10, direction vector of this movie
        trial_coh = ex[1]['tf_trial_coh'].numpy()[0] # len = 10, coh vector of this movie
        true_dirs.append(dom_dir)
        trial_cohs.append(trial_coh)
        start_frames = np.arange(0, 80, 8)
        end_frames = np.arange(4, 84, 8)
        framerate = 60
        for i in range(len(dom_dir)): # 10
            trial = ex[0][:, start_frames[i]*framerate:end_frames[i]*framerate] # [1, 240, 36, 64, 3]
            trials.append(trial)

        # concatenate the trials into a large tensor
        movie = tf.concat(trials, axis=0) # [10, 240, 36, 64, 3]
        movies.append(movie)
        k+=1
    
    all_movies = tf.concat(movies, axis=0) # [300,240,36,64,3]
    print(all_movies.shape)
    # load the CNN model and predict the x and y coordinates
    model_path = HOME_PATH + 'xy_model8'
    cnn_model = keras.models.load_model(model_path)
    # convert xy coors to angles 
    xy_pred = cnn_model.predict(all_movies)
    angs = np.arctan2(xy_pred[1], xy_pred[0])
    # print(angs.reshape((angs.shape[0],)))
    categories = angs_to_categories(angs).reshape((angs.shape[0],))
    # excluding the gray screens
    pred_directions = categories
    true_directions = np.concatenate(true_dirs)
    coherences = np.concatenate(trial_cohs)
    # indices of coherence levels 100%
    coh_ind_100 = np.where(coherences==1)[0]
    coh_ind_15 = np.where(coherences==0)[0]
    # accuracies at different coherence levels
    accuracy100 = (pred_directions[coh_ind_100] == true_directions[coh_ind_100]).mean()
    accuracy15 = (pred_directions[coh_ind_15] == true_directions[coh_ind_15]).mean()
    print(accuracy100, accuracy15)
        
        
if __name__ == '__main__':
    main()



