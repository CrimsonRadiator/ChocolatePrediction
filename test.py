import NeuralNetwork
import bikeDatasetLoader

training_data, result_data = bikeDatasetLoader.readBikeDataSet()

network = NeuralNetwork.Network([12, 15, 1])

print('training data: ' + '\n' + training_data.__str__() + '\n')
network_input = training_data[:-1]
desired_output = training_data[-1]
print(network_input.__len__())
print('network input: ' + '\n' + network_input.__str__() + '\n')
print('desired output: ' + '\n' + desired_output.__str__() + '\n')

for i in range(1, 200):
    #print('weights: ' + n.weights.__str__())
    #print(network.SGD([[network_input, desired_output]], 1, 1, 1))
    pass
