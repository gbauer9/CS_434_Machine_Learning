import numpy as np
import random
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from datetime import datetime


#Loads data in the original format (9 columns, one for each feature)
def load_og_data(filenames):
    #Array of data that will be returned
    data = []
    for filename in filenames:
        #Get .csv into matrix of strings
        temp = np.loadtxt(filename, dtype=str, delimiter=',')


        #Extract the hour from matrix and store in its own array
        hours = [[datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%SZ").hour] for row in temp]
        #Delete hours from main matrix and convert rest of values to floats
        temp = np.delete(temp, 0, 1)
        temp = temp.astype(float)
        #Add hours back to the front of the matrix
        temp2 = np.hstack((hours, temp))


        #Create chunks of either 30 minutes or however long until an episode, then average all those values into one feature vector via make_instance()
        chunk = []

        for row in temp2:
            chunk.append(row)
            if row[-1] == 1 or len(chunk) == 7:
                #Keep class, because we don't want it to be averaged in make_instance()
                cl = row[-1]
                #Only take the first 9 values (the features) and add the class back to the end of the instance
                instance = make_instance(chunk)[:-1]
                instance = np.append(instance, cl)
                data.append(instance)
                #Reset chunk
                chunk = []

    return data

#Loads data that has been flattened (each chunk is a 1x63 array)
def load_flattened_data(filenames):
    data = []

    for filename in filenames:
        temp = np.loadtxt(filename, dtype=float, delimiter=',')

        for row in temp:
            chunk = np.array(row)
            chunk = np.reshape(chunk, (7, 9), order='F')
            instance = make_instance(chunk)
            data.append(instance)

    return data

def make_instance(chunk):
    chunk = np.array(chunk)

    return chunk.mean(0)

def classify_and_write(train, trainClasses, test, outfileName):

    clf = MLPClassifier(hidden_layer_sizes=(2, 4, 6), max_iter=500, random_state=0)

    clf.fit(train, trainClasses)
    pred = clf.predict(test)

    of = open(outfileName, 'w')

    for p in pred:
        of.write(str(p) + '\n')
        if p == 1:
            print('yo')

    of.close()

    return


def main():
    random.seed(2018)

    trainingFiles = ["TrainData/Subject_1.csv", "TrainData/Subject_4.csv", "TrainData/Subject_6.csv", "TrainData/Subject_9.csv"]
    testFiles = ["TestData/general_test_instances.csv"]
    trainData = np.array(load_og_data(trainingFiles))
    testData = np.array(load_flattened_data(testFiles))

    classes = trainData[:, -1]
    trainData = np.delete(trainData, -1, 1)

    classify_and_write(trainData, classes, testData, "general_pred3.csv")

    return

if __name__ == "__main__":
    main()
