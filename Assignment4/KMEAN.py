#K-MEAN
import csv
import numpy as np
import math
import time
from sklearn import preprocessing
import operator
from collections import Counter
import pylab as plt

def norm_ary(file):                             #reads in file, converts to a matix, normalizes and returns the array
    with open(file) as file1 :
        reader = file1.readlines()
        #diagnosis = []
        data = []
        for row in reader:                      #reads in each row
            temp1 = ""
            for t in row:
                temp1 += t
            temp2 = [int(e) if e.isdigit() else e for e in temp1.split(',')]
            stats = np.array(temp2)
            #if len(stats) != 784:
                #print(len(stats))
            #diagnosis.append(stats[0])          #removes and stores diagnosis for normalization
            #stats[0] = 0
            data.append(stats)
        patients = np.vstack(data)
        #print(len(patients))
        norm_stats = preprocessing.normalize(patients)
        #for i in range(len(norm_stats)):        #adds diagnosis back
        #    norm_stats[i][0] = diagnosis[i]
    return norm_stats                           #returns array

def dist(train_point, test_point):              #calculates distance between 2 data points, returns int
    sum = 0
    for i in range(1,len(train_point)):
        sum += pow(train_point[i] - test_point[i],2)
    return math.sqrt(sum)                       #returns int


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

def k_mean(training, clusters):
    for t in training:
        if not (t==clusters[1][0]).all():
            if not (t==clusters[0][0]).all():
                if dist(t, clusters[0][0]) < dist(t, clusters[1][0]):
                    clusters[0].append(t)
                else:
                    clusters[1].append(t)
    return clusters

def nearest(cluster, point):
    min = 10000
    for c in cluster:
        #print("d: ",d)
        #print(min)
        if dist(c, point) < min:
            temp = c
    return temp

def centroid(cluster):
    temp = []
    sum = 0
    for i in range(len(cluster[0])):
        for c in cluster:
            sum += c[i]
        temp.append(sum/len(cluster))
    np.array(temp)
    temp = nearest(cluster, temp)
    return temp

def add_c(cluster, centers):
    centers.append(cluster[1][0])
    centers.append(cluster[0][0])
    return centers

def check(cluster, centers):
    for c in centers:
        if (c==cluster[1][0]).all():
            return False
        if (c==cluster[0][0]).all():
            return False
    return True

train = "data-1.txt"
start_time = time.time()
training = norm_ary(train)
print("--- %s seconds ---" % (time.time() - start_time))
print("upload complete")
print(len(training))
k = 2
clusters = []
temp1 = []
temp1.append(training[12])
clusters.append(temp1)
i = 2
for r in range(k-1):
    temp = []
    temp.append(training[int(len(training)/i)])
    clusters.append(temp)
    i += 1
#print(len(clusters[0][0]))
#print(len(clusters[1][0]))
centers = list(clusters)
temp2 = 0
while abs(len(clusters[0]) - temp2) > .00001:
    start_time1 = time.time()
    clusters = list(k_mean(training, clusters))
    #print("--- %s seconds ---" % (time.time() - start_time1))
    print(len(clusters[0]), len(clusters[1]))

    temp2 = len(clusters[0])
    #start_time2 = time.time()
    clusters = list([[centroid(clusters[0])],[centroid(clusters[1])]])
    print("--- %s seconds ---" % (time.time() - start_time1))
    if check(clusters, centers):
        centers = add_c(clusters, centers)
    else:
        break
    #print(len(clusters[0]), len(clusters[1]))
