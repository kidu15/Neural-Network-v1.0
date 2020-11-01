#perceptron 

import numpy as np 

def sigmoid(x): 
    return 1/ (1 + np.exp(-x))

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T 

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1 

print ('Random starting synaptic weights: ')
print (synaptic_weights)

for iteration in range(1): 

    print ('This is the iteration:')
    print (iteration)
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

#until here, it is trained one time, with no error correction 

print ('Output after training with no error correction: ')
print (outputs) 

#Training process : 
#1. Take the inputs from the training example and put them through the formula to get the neurone's outuput 
#2. Calculate the erro ( difference between the output we got and the actual output)
#3. Depending on the error, adjust the weights accordingly 
#4. Repeat 20k times 


#Error weighted derivative 

def sigmoid_derivative(x): 
    return x * (1 - x) 

for iteration in range(100000): 

    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weight after training ')
print(synaptic_weights)
print('Outputs after training ')
print (outputs)


