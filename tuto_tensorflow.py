# Python 3
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

import filemanager as fm
import csvparser as parser
import learningmachine as lm

# Create the training tf.data.Dataset
model = lm.train(fm.train_dataset)

test_dataset = fm.test_dataset
test_dataset = test_dataset.skip(1)			# skip header row
test_dataset = test_dataset.map(parser.parse_csv)	# parse each row with the function created earlier
test_dataset = test_dataset.shuffle(1000)	# randomize
test_dataset = test_dataset.batch(32)		# use the same batch size as the training set

# Evaluate the model on the test dataset
test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
	prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
	test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

# Use the trained model to make predictions
class_ids = ["Iris setosa", "Iris versicolor", "Iris virginica"]

predict_dataset = tf.convert_to_tensor([
	[5.1, 3.3, 1.7, 0.5],
	[5.9, 3.0, 4.2, 1.5],
	[6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
	class_idx = tf.argmax(logits).numpy()
	name = class_ids[class_idx]
	print("Example {} prediction: {}".format(i, name))

