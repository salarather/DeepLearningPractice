from numpy import exp, array, random, dot


class NeuralNetwork():
	def __init__(self):

		#seed the random generator so it generates same numbers every time the program runs

		random.seed(1)

		# modelling a neuron with 3 input connections and 1 output connection. assign random weights to
		# a 3 x 1 matrix with values in the range of -1 and 1 and a mean of 0

		self.synaptic_weights = 2 * random.random((3,1)) - 1

	#the sigmoid function which describes an s shaped curve that I pass the weighted sum of the inputs
	#through to normalize them between 0 and 1

	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	def __sigmoid_derivative(self, x):

		return x * (1 - x)

	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):

		for iteration in xrange(number_of_training_iterations):

			#pass the training set through neural net
			output = self.predict(training_set_inputs)

			# calculate the error
			error = training_set_outputs - output

			#multiply the error by the input and again by the gradient of the sigmoid curve
			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))


			self.synaptic_weights += adjustment


	def predict(self, inputs):
		# pass inputs through my neural network (my single neuron)
		return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == '__main__':

	#initialize a single neuron neural network

	neural_network = NeuralNetwork()

	print 'Random starting synaptic weights'
	print neural_network.synaptic_weights

	# The training set, we have 4 examples each consisting of 3 input values
	# and 1 output value

	training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
	training_set_outputs = array([[0,1,1,0]]).T

	#train the neural network using a training set
	#have 10,000 iterations

	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print 'New syanptic weights after training: '
	print neural_network.synaptic_weights

	#test the neural network
	print 'hello'
	print 'predicting with new situation [1,0,0]: '
	print neural_network.predict(array([1,0,0])) 
