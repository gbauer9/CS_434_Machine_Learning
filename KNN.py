#KNN
import csv
import numpy as np
import math
from sklearn import preprocessing
import operator
from collections import Counter
import pylab as plt


def plots(stuff):
    stuff

def norm_ary(file):
    with open(file) as file1 :
        reader = csv.reader(file1)
        diagnosis = []
        data = []
        for row in reader:
            stats = np.array(row)
            diagnosis.append(stats[0])          #removes and stores diagnosis for normalization
            stats[0] = 0
            data.append(stats)
        patients = np.vstack(data)
        norm_stats = preprocessing.normalize(patients)
        for i in range(len(norm_stats)):        #adds diagnosis back
            norm_stats[i][0] = diagnosis[i]
    return norm_stats

def dist(train_point, test_point):
    sum = 0
    for i in range(1,len(train_point)):
        sum += pow(train_point[i] - test_point[i],2)
    return math.sqrt(sum)

def nearest_k(train_set, new_point, k, i = None):
    neighbors = []
    if i is None:
        for t in train_set:
            delta_x = dist(t, new_point)
            neighbors.append((t, delta_x))
            #print(neighbors)
        neighbors.sort(key = operator.itemgetter(1))
        nearest = []
        for r in range(k):
            nearest.append(neighbors[r][0][0])
        #print(nearest)
        return(Counter(nearest).most_common(1)[0][0])
    else:
        for t in range(0,len(train_set)):
            if t != i:
                #print("omit")
                delta_x = dist(train_set[t], new_point)
                neighbors.append((train_set[t], delta_x))
                #print(neighbors)
        neighbors.sort(key = operator.itemgetter(1))
        nearest = []
        for r in range(k):
            nearest.append(neighbors[r][0][0])
        #print(nearest)
        return(Counter(nearest).most_common(1)[0][0])

def accuracy(training, testing, k, i = None):
    testing_accuracy = 0
    #accuracy_list = []
    if i is None:
        for t in testing:
            if t[0] == nearest_k(training, t, k):
                #print("correct")
                testing_accuracy += 1
        return testing_accuracy
    else:
        for t in testing:
            if t[0] == nearest_k(training, t, k, i):
                #print("correct")
                testing_accuracy += 1
        return testing_accuracy

test = "knn_test.csv"
train = "knn_train.csv"
training = norm_ary(train)
testing = norm_ary(test)
training_accuracy = 0
testing_accuracy = 0
LOOVC_accuracy = 0
temp = 0
training_error = []
LOOCV_error = []
testing_error = []
x_axis = []
x1_axis =[]
for k in range(1,31,2):
    #print("k: ", k)
    training_error.append(accuracy(training, training, k)/len(training))
    #print("training error ", k, "done")
    testing_error.append(accuracy(training, testing, k)/len(training))
    #print("testing error ", k, "done")
    x_axis.append(k)
print("done with training and testing.")
for k1 in range(1,31,10):
    for i in range(0,len(training)):
        temp1 = accuracy(training, testing, k1, i)
        if temp < temp1:
            temp = temp1
        #print("LOOVC error i: ", i, "k: ", k, "done")
    print("LOOVC error ", k1, "done")
    LOOCV_error.append(temp)
    x1_axis.append(k1)
plt.plot(x_axis, training_error, 'r--', x_axis, testing_error, 'b--', x1_axis, LOOCV_error, 'g--')
plt.show()
#print("accuracy: ", accuracy/len(testing))
