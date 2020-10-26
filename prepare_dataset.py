import os
import cv2
import argparse
import skvideo.io
import numpy as np
import scipy.io
import tensorflow as tf
import blosc

from io import BytesIO

# current working directory
if(os.getcwd()[-1] == '/'):
    cwd = os.getcwd()
else:
    cwd = os.getcwd() + '/'

# changed (width, height) to (64, 36) for compatibility with CNN
def read_video(path, return_frames=False, width=64, height=36, encoding='png'):
    # note that these are the width and height dimensions that I downsized my video frames to.
    # I noticed that your videos seem to be square-framed (unless that's an artefact of how it was saved) -
    # I recommend you to use these rectangular dimesions instead.
    # I also recommend you to have gray screen in between each trial and during the delay period, as opposed to white.
    # Please specify in metadata (if you haven't already) exactly when each trial begins and ends
    # and the time frame in which the network must give a response about the category.
    print("reading video...")

    reader = skvideo.io.vreader(path, num_frames=4800) # remember to specify num frames here

    encoded_list = []
    frame_list = []

    collect_frames = return_frames
    if encoding == 'blosc':
        collect_frames = True

    for frame in reader:
        # the natural movies have 3 channels, so it's better not to change to grayscale here, which has only 1 channel
        # frame = tf.image.rgb_to_grayscale(frame)
        frame = tf.cast(tf.image.resize(frame, (height, width)), tf.uint8)
        # encoded = tf.io.encode_png(frame)
        if encoding == 'png':
            encoded = cv2.imencode('.png', frame.numpy())[1].tobytes()
        elif encoding == 'blosc':
            encoded = None
        else:
            np_bytes = BytesIO()
            np.save(np_bytes, frame)

            encoded = np_bytes.getvalue()
        encoded_list.append(encoded)
        if collect_frames:
            frame_list.append(frame)
    if collect_frames:
        frames = np.array(frame_list)
        if encoding == 'blosc':
            np_bytes = BytesIO()
            np.save(np_bytes, frames)
            encoded_list = [blosc.compress(np_bytes.getvalue())]
    print('reading done')

    if return_frames:
        return encoded_list, frames
    return encoded_list


def _bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def prepare_example(video_path, desc_path, width=64, height=36, debug=False):
    encoded = read_video(video_path, width=width, height=height)

    # what follows is the code used to prepare the metadata from my dot coherence change task
    # for example I specify the frame rate so I can be sure of what is happening each ms
    # I specify the start and end times and frames of the trial along with the coherence change time (if any) and frame
    # change_label reflects whether there was a change at all
    # and label_duration again reflects the frames
    # coh_label reflects whether the starting coherence is 15 or 100%,
    # and therefore what the opposite coherence is after the change (if any) on a given trial
    # I also have the direction (out of four) in which the dots are drifting

    desc = scipy.io.loadmat(desc_path)
    frame_rate = desc['framerate'][0, 0] # ~60fps here
    grey_dur = desc['grey_dur'][0, 0] # duration of grey screens
    # start and end times of change trials
    start_times = np.cumsum(np.concatenate(([0], desc['trialend'][0, :-1] + grey_dur)))
    end_times = np.cumsum(desc['trialend'][0] + grey_dur) - grey_dur
    # start and end frames
    start_frames = (start_times * frame_rate).astype(np.int)
    end_frames = (end_times * frame_rate).astype(np.int)
    # start and end of each grey screen
    grey_start_times = end_times
    grey_end_times = np.append(start_times[1:], 80)
    # start and end frames of each grey screen
    grey_start_frames = (grey_start_times * frame_rate).astype(np.int)
    grey_end_frames = (grey_end_times * frame_rate).astype(np.int)

    # change and coh and directions
    change_times = desc['changetimes'][0]
    trial_coh = desc['trialcoh'][0] # 0 for start with 15%, 1 for start with 100%
    dominant_direction = (desc['trialdir'][0] / 90).astype(np.int) + 1
    is_changes = change_times > 0
    change_frames = ((start_times + change_times) * frame_rate).astype(np.int)
    change_label = np.zeros(len(encoded), dtype=np.int64)
    label_duration = int(.4 * frame_rate)
    coh_label = np.zeros(len(encoded), dtype=np.int64)
    direction_label = np.zeros(len(encoded), dtype=np.int64)
    print("Number of frames of each movie: ", len(direction_label))

    for is_change, change_frame in zip(is_changes, change_frames):
        change_label[change_frame:change_frame + label_duration] = is_change

    for i in range(len(trial_coh)):
        coh_label[start_frames[i]:end_frames[i]] = trial_coh[i]
        if is_changes[i]:
            coh_label[change_frames[i]:end_frames[i]] = 1 - trial_coh[i]
        direction_label[start_frames[i]:end_frames[i]] = dominant_direction[i]

    features = {
        # this is how I would save a dictionary of my coherence change metadata
        'frames': _bytes_list_feature(encoded), # this is the encoded movie frames
        'change_label': _int64_list_feature(change_label),
        'coherence_label': _int64_list_feature(coh_label),
        'direction_label': _int64_list_feature(direction_label),
        'dominant_direction': _int64_list_feature(dominant_direction),
        'trial_coherence': _int64_list_feature(trial_coh),
        'start_frames': _int64_list_feature(start_frames), # will need the start and end frames when we slice the movie into trials
        'end_frames': _int64_list_feature(end_frames),
        'grey_start_frames': _int64_list_feature(grey_start_frames),
        'grey_end_frames': _int64_list_feature(grey_end_frames),
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    if not debug:
        return example
    # for coherence change task
    return example, [encoded, change_label, 
                     coh_label, direction_label, 
                     dominant_direction, trial_coh,
                     start_frames, end_frames,
                     grey_start_frames, grey_end_frames]


def main():
    parser = argparse.ArgumentParser(description='Args for processing videos')
    parser.add_argument('root_path', type=str, help='root directory of videos (which includes also metadata)')
    parser.add_argument('root_path_metadata', type=str, help='root directory of videos (which includes also metadata)')
    parser.add_argument('from_index', type=int, help='process all trials starting from this index')
    parser.add_argument('to_index', type=int, help='process all trials until (exclusive) this index')
    # parser.add_argument('save_to', type=str, help='output path for tfrecord file') # for coherence change task, this was /home/macleanlab/stim/coh_change/preprocessed/
    args = parser.parse_args()

    for i in range(args.from_index, args.to_index):
        writer = tf.io.TFRecordWriter(cwd+f'preprocessed/processed_data_{i}.tfrecord')
        video_path = os.path.join(args.root_path, f'stim_{i}.mov')
        desc_path = os.path.join(args.root_path_metadata, f'stim_{i}_info.mat')
        example = prepare_example(video_path, desc_path)
        writer.write(example.SerializeToString())
        print('>', end='')
    print()
    writer.close()
    print(f'= processed {args.to_index - args.from_index} video sequences successfully =')


def debug():
    video_path = os.path.expanduser('~/stim_000001.mov')
    desc_path = os.path.expanduser('~/info_000001.mat')
    encoded, frames = read_video(video_path, return_frames=True)

    desc = scipy.io.loadmat(desc_path)

    # similar to in prepare_example, which reflects the metadata for the dot coherence change task
    # you will change accordingly, for how you desire to debug

    frame_rate = desc['framerate'][0, 0]
    start_times = np.cumsum(np.concatenate(([0], desc['trialend'][0, :-1] + 3)))
    change_times = desc['changetimes'][0]
    is_changes = change_times > 0
    change_frames = ((start_times + change_times) * frame_rate).astype(np.int)
    change_label = np.zeros(frames.shape[0], dtype=np.bool)
    label_duration = int(.4 * frame_rate)
    for is_change, change_frame in zip(is_changes, change_frames):
        change_label[change_frame:change_frame + label_duration] = is_change

    x = np.arange(frames.shape[0]) / frame_rate
    time_avg = frames.mean((1, 2, 3))
    ch = change_label * time_avg.mean()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, time_avg)
    ax.plot(x, ch)
    plt.show()


def debug_new():
    video_path = os.path.expanduser('~/stim_000001.mov')
    desc_path = os.path.expanduser('~/info_000001.mat')
    ex, li = prepare_example(video_path, desc_path, debug=True)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(li[-1])
    ax.plot(li[-2])
    ax.plot(li[-3])
    plt.show()


def compare_codecs():
    # this is the path to all your fullres .mov files
    video_path = os.path.expanduser('~/data/yuqing/fullres/stim_000001.mov')
    vv, frames = read_video(video_path, return_frames=True, width=200, height=125, encoding='blosc')

    import pudb
    pudb.set_trace()


def check_filter():
    video_path = os.path.expanduser('~/stim_1501.mov')
    vv, frames = read_video(video_path, return_frames=True, width=200, height=125, encoding='png')

    filtered_frames = np.zeros_like(frames)
    last_frame = filtered_frames[0]
    decay = .94
    for i in range(frames.shape[0]):
        filtered_frames[i] = decay * last_frame + (1 - decay) * frames[i]
        last_frame = filtered_frames[i]
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, figsize=(6, 8))
    axes[0].imshow(filtered_frames[850], cmap='gray')
    axes[1].imshow(filtered_frames[950], cmap='gray')
    [ax.axis('off') for ax in axes]
    fig.tight_layout()
    fig.savefig('stimulus.png', dpi=300)
    print('filtered')


if __name__ == '__main__':
    main()
