import NeuralNetwork
import bikeDatasetLoader

training_data, result_data = bikeDatasetLoader.readBikeDataSet()

network = NeuralNetwork.Network([12, 15, 1])
