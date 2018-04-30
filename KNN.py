#KNN
import csv
import numpy as np
import math
from sklearn import preprocessing
import operator
from collections import Counter
import pylab as plt
"""Simply run the program to have it display the plot.  Finding the LOOCV error for every value of k takes a very long time
and I would recommend increasing the increment on the for loop"""

def norm_ary(file):                             #reads in file, converts to a matix, normalizes and returns the array
    with open(file) as file1 :
        reader = csv.reader(file1)
        diagnosis = []
        data = []
        for row in reader:                      #reads in each row
            stats = np.array(row)
            diagnosis.append(stats[0])          #removes and stores diagnosis for normalization
            stats[0] = 0
            data.append(stats)
        patients = np.vstack(data)
        norm_stats = preprocessing.normalize(patients)
        for i in range(len(norm_stats)):        #adds diagnosis back
            norm_stats[i][0] = diagnosis[i]
    return norm_stats                           #returns array

def dist(train_point, test_point):              #calculates distance between 2 data points, returns int
    sum = 0
    for i in range(1,len(train_point)):
        sum += pow(train_point[i] - test_point[i],2)
    return math.sqrt(sum)                       #returns int

def nearest_k(train_set, new_point, k, i = None):       #finds nearest k neighbors, has them vote and returns the winning vote
    neighbors = []
    if i is None:                                       #when not using leave on out cross validation
        for t in train_set:
            delta_x = dist(t, new_point)
            neighbors.append((t, delta_x))
        neighbors.sort(key = operator.itemgetter(1))    #sorts neigbors into order accourding to distance
        nearest = []
        for r in range(k):
            nearest.append(neighbors[r][0][0])
        return(Counter(nearest).most_common(1)[0][0])   #returns int
    else:                                               #when using leave on out cross validation
        counter = 0
        for t in train_set:
            if counter != i:
                delta_x = dist(t, new_point)
                neighbors.append((t, delta_x))
            counter += 1
        neighbors.sort(key = operator.itemgetter(1))    #sorts neigbors into order accourding to distance
        nearest = []
        for r in range(k):
            nearest.append(neighbors[r][0][0])
        return(Counter(nearest).most_common(1)[0][0])   #returns int

def accuracy(training, testing, k, i = None):           #calculates accuracy of method, returns float
    testing_accuracy = 0
    if i is None:                                       #for not using leave one out cross validation
        for t in testing:
            if t[0] == nearest_k(training, t, k):
                testing_accuracy += 1
        return testing_accuracy                         #returns float
    else:                                               #for not using leave one out cross validation
        for t in testing:
            if t[0] == nearest_k(training, t, k, i):
                testing_accuracy += 1
        return testing_accuracy                         #returns float

test = "knn_test.csv"
train = "knn_train.csv"
training = list(norm_ary(train))
testing = list(norm_ary(test))
training_accuracy = 0
testing_accuracy = 0
LOOVC_accuracy = 0
temp = 0
training_error = []
LOOCV_error = []
testing_error = []
x_axis = []
x1_axis =[]
for k in range(1,53,2):                                                 #for testing and training error at various k values
    training_error.append(accuracy(training, training, k)/len(training))
    testing_error.append(accuracy(training, testing, k)/len(training))
    x_axis.append(k)
for k1 in range(1,53,7):                                                #for leave one out cross validation testing error at various k values
    for i in range(0,len(training)):
        temp1 = accuracy(training, testing, k1, i)
        if temp < temp1:
            temp = temp1
    LOOCV_error.append(temp/len(training))
    x1_axis.append(k1)
plt.plot(x_axis, training_error, 'r--', x_axis, testing_error, 'b--', x1_axis, LOOCV_error, 'g--')  #plots training, testing, and LOOCV errors of various k values
plt.show()                                                                                          #shows above plot
print(training_error)
print(testing_error)
print(LOOCV_error)
