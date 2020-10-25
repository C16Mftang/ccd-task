import os
import cv2
import argparse
import skvideo.io
import numpy as np
import scipy.io
import tensorflow as tf
# import blosc

from io import BytesIO

# current working directory
if(os.getcwd()[-1] == '/'):
    cwd = os.getcwd()
else:
    cwd = os.getcwd() + '/'

# root_path = '/home/macleanlab/stim/coh_change/fullres/fullres'
root_path = '/Users/tangmufeng/Desktop/UChicago/Courses/MacleanLab_research/tfrecord_data_processing/fullres'
for i in range(1):
    print(f'Resizing movie {i+2}...')
    path = os.path.join(root_path, f'stim_{i+2}.mov')
    reader = skvideo.io.vreader(path)
    frames = []
    for frame in reader:
        frame = tf.cast(tf.image.resize(frame, (36, 64)), tf.uint8)
        frame = frame.numpy()
        frame = np.expand_dims(frame, axis=0)
        frames.append(frame)
    movie = np.concatenate(frames, axis=0)
    print(movie.shape)

    skvideo.io.vwrite(f'resized_stim_{i+2}.mp4', movie)