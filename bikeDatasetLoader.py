import csv
import numpy as np
import random
"""
Library for loading Bike-Sharing-Dataset data. 
"""

"""
Normalizes csv row parameters to fit [0, 1] range which is ideal for neural network.
"""
def normalizeInputDataTable(dataTable):
    return [[(float(row[2])-1)/3, float(row[3]), (float(row[4])-1)/11, (float(row[5]))/23,
            float(row[6]), (float(row[7]))/6, float(row[8]), (float(row[9])-1)/3, float(row[10]),
            float(row[11]), float(row[12]), float(row[13])] for row in dataTable]


"""
This function opens hour.csv file from Bike-Sharing-Dataset and returns
tuple containing training and testing data in such manner as ndarray:
resultTrainingData[0] - arrays of inputs
resultTrainingData[1] - outputs
"""
def readBikeDataSet():
    #open csv file
    with open('Bike-Sharing-Dataset/hour.csv', newline='') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',', quotechar='"')
        #read number of rows
        row_count = sum(1 for row in csvReader)
        #come back at beginning of file
        csvFile.seek(0)
        #generate random indexes of testing data - it will be 10% of total data
        testDataIndexes = random.sample(range(1, row_count), int(row_count/10))
        #initialize data lists
        testData = []
        trainingData = []

        #ommit first row with info data
        firstLine = next(csvReader)

        idx = 1
        max = 0.0
        #distribute data between test and training lists
        for row in csvReader:
            # print(row)
            if float(row[16]) > max:
                max = float(row[16])
            if idx in testDataIndexes:
                testData.append(row)
            else:
                trainingData.append(row)
            idx = idx + 1

        # print(max)

        #return tuples of training and test data
        resultTrainingData = (np.array(normalizeInputDataTable(trainingData)), np.array([float(row[16])/max for row in trainingData]))
        resultTestData = (np.array(normalizeInputDataTable(testData)), np.array([float(row[16])/max for row in testData]))

        return (resultTrainingData, resultTestData)


if __name__ == "__main__":
        readBikeDataSet()