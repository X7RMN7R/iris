# Python 3
# Learning machine
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import csvparser as parser

def loss(model, x, y):
	y_ = model(x)
	return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
	
def grad(model, inputs, targets):
		with tf.GradientTape() as tape:
			loss_value = loss(model, inputs, targets)
		return tape.gradient(loss_value, model.variables)
		
def train(train_dataset):
	train_dataset = train_dataset.skip(1)					# skip the first header row
	train_dataset = train_dataset.map(parser.parse_csv)		# parse each row
	train_dataset = train_dataset.shuffle(buffer_size=1000)	# randomize
	train_dataset = train_dataset.batch(32)
	
	# View a single example entry from a batch
	features, label = iter(train_dataset).next()
	print("example features:", features[0])
	print("example label:", label[0])
	
	# Create a model
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),	# input shape required
		tf.keras.layers.Dense(10, activation="relu"),
		tf.keras.layers.Dense(3)
	])

	# Create an optimizer
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

	# training loop
	## Note : rerunning this cell uses the same model variables

	# keep results for plotting
	train_loss_results = []
	train_accuracy_results = []

	num_epochs = 201

	for epoch in range(num_epochs):
		epoch_loss_avg = tfe.metrics.Mean()
		epoch_accuracy = tfe.metrics.Accuracy()

		# Training loop - using batches of 32
		for x, y, in train_dataset:
			# Optimize the model
			grads = grad(model, x, y)
			optimizer.apply_gradients(zip(grads, model.variables),
						global_step=tf.train.get_or_create_global_step())

			# Track progress
			epoch_loss_avg(loss(model, x, y)) 	# add current batch loss
			# compare predicted label to actual label
			epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

		#end epoch
		train_loss_results.append(epoch_loss_avg.result())
		train_accuracy_results.append(epoch_accuracy.result())

		if epoch % 50 == 0:
			print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
										epoch_loss_avg.result(),
										epoch_accuracy.result()))
									
	return model
