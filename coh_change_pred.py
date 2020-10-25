import tfrecord_dataset
from absl import flags
from absl import app
import os
from keras import Model, Input
import keras

# current working directory
if(os.getcwd()[-1] == '/'):
    cwd = os.getcwd()
else:
    cwd = os.getcwd() + '/'

HOME_PATH = '/home/macleanlab/mufeng/models/'

FLAGS = flags.FLAGS

# step one: define necessary flags
flags.DEFINE_integer('max_seq_len', 3500, '') # max duration of each trial in ms
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('n_epochs', 100, '')
# data path may change, currently it is here:
flags.DEFINE_string('data_path',cwd+'preprocessed/','')


def main(argv):
    train_filenames = [os.path.join(FLAGS.data_path, f'processed_data_{i}.tfrecord') for i in range(30)]
    test_filenames = [os.path.join(FLAGS.data_path, f'processed_data_{i}.tfrecord') for i in range(30, 40)]
    data_set = tfrecord_dataset.create_dataset(train_filenames, FLAGS.max_seq_len, pool=None).batch(FLAGS.batch_size)
    # for test set:
    test_data_set = tfrecord_dataset.create_dataset(test_filenames, FLAGS.max_seq_len).batch(FLAGS.batch_size)

    model_path = HOME_PATH + 'xy_model8'
    cnn_model = keras.models.load_model(model_path)
    for ex in test_data_set:
      y_pred = cnn_model.predict(ex[0])
      print(y_pred.shape)

if __name__ == '__main__':
    app.run(main)