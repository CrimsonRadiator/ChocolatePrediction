import NeuralNetwork
import math
import datasetLoader


training_data, result_data = datasetLoader.readConcreteDataset()

network = NeuralNetwork.Network([8, 10, 20, 1])

paired_data = []
paired_result_data = []

for x_row, y_row in zip(training_data[0], training_data[1]):
    paired_data.append([x_row.reshape(8, 1), y_row.reshape(1, 1)])

for x_row, y_row in zip(result_data[0], result_data[1]):
    paired_result_data.append([x_row.reshape(8, 1), y_row.reshape(1, 1)])

network.SGD(paired_data, 20, 0.1, 1000, paired_result_data)


total_error = 0
for row in paired_result_data:
    print(row[1], network.feedforward(row[0]))
    tmp = abs(row[1] - network.feedforward(row[0]))/row[1]
    total_error+=tmp
print(total_error/len(paired_result_data))

