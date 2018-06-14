import numpy as np
import random
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from datetime import datetime


#Loads data in the original format 
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

def make_instance(chunk):
    chunk = np.array(chunk)

    return chunk.mean(0)    

def load_flattened_data(filename):
    temp = np.loadtxt(filename, dtype=float, delimiter=',')
    
    for row in temp:
        temp2 = np.array(row)
        temp2 = np.reshape(temp2, (7, 9))
        print(temp2)
    

def main():
    random.seed(2018)

    trainingFiles = ["Subject_1.csv", "Subject_4.csv", "Subject_6.csv", "Subject_9.csv"]
    testFile = "general_test_instances.csv"
    trainData = np.array(load_og_data(trainingFiles))
    testData = np.array(load_flattened_data(testFile))

    classes = trainData[:, -1]
    trainData = np.delete(trainData, -1, 1)

    bestIndices = ()
    maxF1 = 0

    '''
    for i in range(1, 10):
        for j in range(1, 10):
            for w in range(1, 10):
                clf = MLPClassifier(hidden_layer_sizes=(i, j, w), max_iter=300)

                clf.fit(data, classes)
                pred = clf.predict(data)

                f1 = f1_score(classes, pred)
                
                if f1 > maxF1:
                    maxF1 = f1
                    bestIndices = (i, j, w, y)

        print("Iteration: " + str(i))

    '''
    clf = MLPClassifier(hidden_layer_sizes=(8, 4, 6), max_iter=500, random_state=0)

    clf.fit(trainData, classes)
    pred = clf.predict(trainData)

    maxF1 = f1_score(classes, pred)
    
    print("MaxF1: " + str(maxF1) + "\nIndices: " + str(bestIndices))
    return

if __name__ == "__main__":
    main()