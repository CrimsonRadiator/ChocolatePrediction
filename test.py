import NeuralNetwork
import bikeDatasetLoader
import numpy

training_data, result_data = bikeDatasetLoader.readBikeDataSet()

network = NeuralNetwork.Network([10, 5, 1])

paired_data = []
paired_result_data = []

for x_row, y_row in zip(training_data[0], training_data[1]):
    paired_data.append([x_row.reshape(10, 1), y_row.reshape(1, 1)])

for x_row, y_row in zip(result_data[0], result_data[1]):
    paired_result_data.append([x_row.reshape(10, 1), y_row.reshape(1, 1)])

network.SGD(paired_data, 10, 0.5, 5)

tmp = 0

for row in paired_data:
    # continue
    #print(row[0])
    if ((row[1] - network.feedforward(row[0])) * 977.0) > 0:
        tmp += (row[1] - network.feedforward(row[0])) * 977.0
    else:
        tmp -= (row[1] - network.feedforward(row[0])) * 977.0
    print('result ', network.feedforward(row[0])*977.0, '\t', row[1]*977.0)

print('\nmean: ', tmp/paired_data.__len__())
