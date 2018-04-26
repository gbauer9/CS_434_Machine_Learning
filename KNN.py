#KNN
import csv
import numpy as np
import math
from sklearn import preprocessing
import operator
from collections import Counter
#import statistics

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

def nearest_k(train_set, new_point, k):
    neighbors = []

    for t in train_set:
        delta_x = dist(t, new_point)
        neighbors.append((t, delta_x))
        #print(neighbors)
    neighbors.sort(key = operator.itemgetter(1))
    nearest = []
    for r in range(k):
        nearest.append(neighbors[r][0][0])
    return(Counter(nearest).most_common(1)[0][0])
    #return (statistics.mode(nearest))


test = "knn_test.csv"
train = "knn_train.csv"
training = norm_ary(train)
testing = norm_ary(test)
k = 51

accuracy = 0
for t in testing:
    if t[0] == nearest_k(training, t, k):
        #print("correct")
        accuracy += 1
print("accuracy: ", accuracy/len(testing))
