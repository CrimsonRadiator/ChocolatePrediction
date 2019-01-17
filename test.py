import NeuralNetwork
import math
import datasetLoader

training_data, result_data = datasetLoader.readBikeDataSet()
#training_data, result_data = datasetLoader.getSinDataSet(1000)

network = NeuralNetwork.Network([11, 50, 1])

paired_data = []
paired_result_data = []

for x_row, y_row in zip(training_data[0], training_data[1]):
    paired_data.append([x_row.reshape(11, 1), y_row.reshape(1, 1)])

for x_row, y_row in zip(result_data[0], result_data[1]):
    paired_result_data.append([x_row.reshape(11, 1), y_row.reshape(1, 1)])

network.SGD(paired_data, 20, 0.5, 100)

tmp = 0

for row in paired_result_data:
    # continue
    # print(row[0])
    tmp += math.fabs((row[1] - network.feedforward(row[0]))) * 977.0
    print('tmp ', tmp, ' result ', network.feedforward(row[0])*977.0, '\t', row[1]*977.0)

print('\nmean: ', tmp/paired_result_data.__len__())
