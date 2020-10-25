import os
import io
import blosc
import numpy as np
import tensorflow as tf
import keras

HOME_PATH = '/home/macleanlab/mufeng/models/'
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

            _frames = tf.py_function(func=fn, inp=[_x['frames'].values[0]], Tout=tf.uint8)[:max_seq_len]
            # _frames = tf.zeros((max_seq_len, 125, 200, 1), tf.uint8)

        # understand this bunch
        # m1: whether randnum > 0.5, if so = 1 otherwise = -1
        _m1 = tf.cast(tf.random.uniform(minval=0, maxval=1, shape=()) > .5, tf.int32) * 2 - 1
        _m2 = tf.cast(tf.random.uniform(minval=0, maxval=1, shape=()) > .5, tf.int32) * 2 - 1
        _frames = _frames[:, ::_m1, ::_m2]
        _seq_len = tf.shape(_frames)[0]
        _p1 = [0, max_seq_len - _seq_len]
        _p = [_p1, [0, 0], [0, 0], [0, 0]]
        # tf.pad here: add (max_seq_len-seq_len) 0's after the contents in the first dimension of frames
        _frames = tf.pad(_frames, _p)

        _label = tf.pad(_x['coherence_label'].values[:max_seq_len], [_p1])
        _change_label = tf.pad(_x['change_label'].values[:max_seq_len], [_p1])
        _change_label += tf.pad(_change_label[:-23], [[23, 0]])
        _direction_label = tf.pad(_x['direction_label'].values[:max_seq_len], [_p1])
        _dominant_direction = _x['dominant_direction'].values[:max_seq_len]
        _trial_coherence = _x['trial_coherence'].values[:max_seq_len]
        #_label += tf.pad(_label[:-23*2], [[23*2, 0]])
        # _x = {'frames': _frames, 'label': _label}
        return _frames, dict(tf_op_layer_coherence=_label, 
                             tf_op_layer_change=_change_label, 
                             tf_op_dir=_direction_label, 
                             tf_dom_dir=_dominant_direction,
                             tf_trial_coh=_trial_coherence)

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
    file_names = [os.path.expanduser(f'preprocessed/processed_data_{i+1}.tfrecord') for i in range(10)]
    data_set = create_dataset(file_names, 3500).batch(1)

    # ex[0] the frames (tensor), shape [1, 3500, 36, 64, 3]
    # ex[1] a dictionary, containing coh level, change label and direction
    k = 1
    """
    we want each movie to be represented by a list of trials, each trial to be represented by a 
    4d tensor of shape (batch,240,36,64,3); for each trial in this list, pass it through the CNN.
    First, need to pad each trial to 240, with 0 (or with something else - could try later)
    """
    movies = []
    true_dirs = []
    trial_cohs = []
    max_len = 240 # temporal dimension compatible with the CNN
    for ex in data_set:
        trials = []
        print("Movie ", k)
        # the direction vector, of length max_seq_len, fixed for each trial/gray screen in the movie
        dirs = ex[1]['tf_op_dir'].numpy()[0]
        dom_dir = ex[1]['tf_dom_dir'].numpy()[0]
        trial_coh = ex[1]['tf_trial_coh'].numpy()[0]
        true_dirs.append(dom_dir)
        trial_cohs.append(trial_coh)
        # detect where the movie switches between trial and rest (vice versa) 
        # i.e. where the next direction value is different from the current one
        switches = list(np.where(dirs[:-1] != dirs[1:])[0])
        s1 = [0] + [k+1 for k in switches]
        s2 = switches + [3500]
        for i in range(len(s1)):
            # split trials and gray screens by indices where direction changes
            trial = ex[0][:, s1[i]:s2[i]+1] 
            seq_len = tf.shape(trial)[1]
            # if the number of frames is < 240, fill the gap with the first frames of the trial
            if seq_len < max_len and seq_len >= max_len//2:
                trial = tf.concat([trial, trial[:, 0:max_len-seq_len]], axis=1)
            elif seq_len < max_len//2:
                padding = [[0, 0], [0, max_len-seq_len], [0, 0], [0, 0], [0, 0]]
                trial = tf.pad(trial, padding)
            # the final frames are typically larger than 240 - but they are gray screens anyway
            else:
                trial = trial[:, :max_len]
            trials.append(trial)

        # concatenate the trials into a large tensor
        movie = tf.concat(trials, axis=0)
        movies.append(movie)
        k+=1
    
    all_movies = tf.concat(movies, axis=0)
    # load the CNN model and predict the x and y coordinates
    model_path = HOME_PATH + 'xy_model8'
    cnn_model = keras.models.load_model(model_path)
    # convert xy coors to angles 
    xy_pred = cnn_model.predict(all_movies)
    angs = np.arctan2(xy_pred[1], xy_pred[0])
    # print(angs.reshape((angs.shape[0],)))
    categories = angs_to_categories(angs).reshape((angs.shape[0],))
    # excluding the gray screens
    pred_directions = categories[::2]
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



