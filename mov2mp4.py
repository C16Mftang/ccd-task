"""
Convert .mov files to .mp4 files
"""

import os
import cv2
import argparse
import skvideo
# path = "C:/ffmpeg/bin"
# skvideo.setFFmpegPath(path)
import skvideo.io
import numpy as np
import scipy.io
import tensorflow as tf
# import blosc
import matplotlib.pyplot as plt

from io import BytesIO

# current working directory
if(os.getcwd()[-1] == '/'):
    cwd = os.getcwd()
else:
    cwd = os.getcwd() + '/'

root_path = 'fullres'
root_path_metadata = 'fullres/trialinfo'
for i in range(30):
    print(f'Resizing movie {i+1}...')
    video_path = os.path.join(root_path, f'stim_{i+1}.mov')
    desc_path = os.path.join(root_path_metadata, f'stim_{i+1}_info.mat')
    desc = scipy.io.loadmat(desc_path)
    framerate = int(round(desc['framerate'][0,0]))
    # reader is an ndarray if use vread, a generator if use vreader
    reader = skvideo.io.vread(video_path, num_frames=4800)

    frames = []
    for j in range(len(reader)):
    # for frame in reader:
        frame = reader[j]
        frame = tf.cast(tf.image.resize(frame, (36, 64)), tf.uint8)
        frame = frame.numpy()
        frame = np.expand_dims(frame, axis=0)
        frames.append(frame)
    movie = np.concatenate(frames, axis=0)
    print(len(frames))
    skvideo.io.vwrite(f'/home/macleanlab/mufeng/NaturalMotionCNN/Movies_dot/stim{i+1}.mp4', movie, inputdict={'-r': '60'}, outputdict={'-r': '60'})

