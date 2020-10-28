# tfrecord_data_processing

This README file tells you how to use scripts in this repo to:
1) translate movie files into tfrecord files
2) load in and decode the tfrecord data to be used for model training

The processed tfrecord data should live on one of the GPU servers: macleanlab@205.208.22.225
Please place all tfrecord dmc stimulus data in this folder on the server: /home/macleanlab/stim/dmc
The folder /home/macleanlab/stim/coh_change contains the stimulus set for the dot coherence change task.

There is a separate README within coh_change subfolder that describes that dataset.
Some of that information, which generalizes to all stimulus sets, is repeated below.

The original videos for the stimulus set was generated using Psychtoolbox in Matlab.
The resulting videos were each very large, and the dataset itself was extensive.
Even after downsizing to desired dimensions, videos are so large
(with dimensions of batch_size x number_frames x height x width)
that loading them is time-intensive and they quickly blow up disc space in numpy arrays.

The solution is for each frame to be encoded as a png and stored in .tfrecord files.

1) TRANSLATE MOVIE FILES INTO TFRECORD FORMAT

prepare_dataset.py loads a number of videos, preprocesses them, and stores them into a tfrecord file.
main() iteratively calls prepare_example to process each video (using read_video function) and its metadata.
You will need to change some parts of the script to reflect your own videos, their metadata and trial structure / how you intend to use metadata to define loss.
I have commented the areas you will most likely need to tailor for your own task. 

prepare_all.py is a helper that looks for all available videos in a certain directory, and then creates multiple tfrecord files at once by calling prepare_dataset.py.

2) LOAD AND DECODE TFRECORD DATA

The script tfrecord_dataset.py loads these .tfrecord files and provides appropriately decoded data for training.
Again you will need to change some parts of the script to reflect your own videos and their metadata.
Under its function create_dataset(), the function _parse_example(_x) decodes the png frames in the tfrecord files.
It then becomes a numpy array of shape number_frames x height x width that is ready for feeding into the model.

You can modify the following code snippets in your own tensorflow scripts to load and decode the data:"""

import tfrecord_dataset

#### step one: define necessary flags
absl.flags.DEFINE_integer('max_seq_len', 3500, '') # max duration of each trial in ms
absl.flags.DEFINE_integer('batch_size', 32, '')
absl.flags.DEFINE_integer('n_epochs', 100, '')
#### data path may change, currently it is here:
absl.flags.DEFINE_string('data_path','/home/macleanlab/stim/dmc/preprocessed/','')

#### step two: define training and test sets
#### example if files with indices 0 - 29 contain the training data
train_filenames = [os.path.join(flags.data_path, f'processed_data_{i}.tfrecord') for i in range(30)]
#### example if files with indices 30 - 39 contain the testing data
test_filenames = [os.path.join(flags.data_path, f'processed_data_{i}.tfrecord') for i in range(30, 40)]

#### step three: load and decode datasets
#### for training set:
data_set = tfrecord_dataset.create_dataset(train_filenames, flags.max_seq_len, pool=None)
#### for test set:
test_data_set = tfrecord_dataset.create_dataset(test_filenames, flags.max_seq_len)
#### you can certainly do this step more complexly, perhaps you want to shuffle the data, create a separate visualization set, etc. etc.
#### check out the create_dataset function in tfrecord_dataset itself to understand its workings better along with its defaults

#### step four: train and test model - the particulars will vary, but you may do something like the following:
model.fit(data_set, epochs=flags.n_epochs, validation_data=test_data_set)
model.evaluate(test_data_set)
