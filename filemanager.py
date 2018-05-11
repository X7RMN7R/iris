# Python 3
# File Manager
import tensorflow as tf
import os

# Download dataset 
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
											origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

train_dataset = tf.data.TextLineDataset(train_dataset_fp)

# Setup the test dataset
test_url = "http://download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
				origin=test_url)

test_dataset = tf.data.TextLineDataset(test_fp)

