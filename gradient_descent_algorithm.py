from numpy import *

# compute mean squared error
def compute_errors(b, m, data):
	total_error = 0

	for i in range(0, len(data)):
		# features
		x = data[i, 0]
		# output
		y = data[i, 1]
		total_error += (y - ((m * x) + b)) ** 2 

	# mean squared error
	return total_error / float(len(data))

# gradient descent algorithm
def compute_gradient_descent(b_current, m_current, data, learning_rate):
	# initial gradient for bias
	b_gradient = 0
	# initial gradient for slope 
	m_gradient = 0
	# total number of data
	N = float(len(data))

	for i in range(0, len(data)):
		# features
		x = data[i, 0]
		# output
		y = data[i, 1]

		b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient) 

	return [new_b, new_m]

# self update bias and slope
def gradient_descent_runner(initial_b, initial_m, data, learning_rate, total_epochs):
	b = initial_b
	m = initial_m
	
	for i in range(total_epochs):
		# self update bias and slope
		b, m = compute_gradient_descent(b, m, array(data), learning_rate)

	return [b, m]

def run():
	# load data
	data = genfromtxt('data.csv', delimiter=",")

	# set learning rate
	learning_rate = 0.0001

	initial_b = 0
	initial_m = 0

	total_epochs = 1000

	print("Initial gradient, m: {0}, Inititial bias, b: {1}, Initial error, E: {2}".format(initial_m, initial_b, compute_errors(initial_m, initial_b, data)))

	print("Running gradient descent algorithm...")
	[b, m] = gradient_descent_runner(initial_b, initial_m, data, learning_rate, total_epochs)

	print("Total epochs: {0}, Bias, b: {1}, Slope, m: {2}, Error, E: {3}".format(total_epochs, b, m, compute_errors(b, m, data)))


# main function
if __name__ == "__main__":
	run()




