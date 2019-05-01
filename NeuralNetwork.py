from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

        # La funcion Sigmoid, que describe una curva en forma de S
        # Se pasan la suma ponderada de las entradas a traves de esta funcion para
        # normalizarlos entre 0 y 1.

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # La derivada de la funcion Sigmoid.
    # Este es el gradiente de la funcion Sigmoid.
    # Indica la confianza que tenemos en el peso existente.

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Entrenamos a la red neuronal a traves de un proceso de prueba y error
    # Se realiza un ajuste de los pesos sinapticos cada vez.

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pasar el conjunto de entrenamiento a traves de nuestra red neuronal (una sola neurona)
            output = self.think(training_set_inputs)

            # Calcular el error(La diferencia entre el resultado deseado
            # y el resultado obtenido).
            error = training_set_outputs - output

            # Multiplica el error por la entrada y nuevamente por el gradiente de la curva Sigmoid
            # Esto significa que los pesos menos confiables se estan ajustando mas
            # Esto significa que las entradas que son cero, no causan cambios en los pesos.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Ajuste de los pesos
            self.synaptic_weights += adjustment

    # La red neuronal piensa.
    def think(self, inputs):
        # Pasar las entradas a traves de nuestra red neuronal (una neurona)
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":
    # Iniciar una red neuronal de una neurona
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # El conjunto de pruebas. Tenemos cuatro ejemplos, cada uno consiste en 3 valores de entrada
    # y un valor de salida.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Entrenar la red neuronal utilizando el conjunto de entrenamiento.
    # Realizar 10000 veces y realizar un ajuste mas pequeno cada vez.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    # Prueba la red neuronal con una nueva situacion
    print "Considering new situation [1, 0, 0] -> ?: "
    print neural_network.think(array([1, 0, 0]))