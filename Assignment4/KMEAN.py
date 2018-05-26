#K-MEAN
import csv
import numpy as np
import math
import time
import random

from sklearn import preprocessing
import operator
from collections import Counter
import pylab as plt

def norm_ary(file):                             #reads in file, converts to a matix, normalizes and returns the array
    with open(file) as file1 :
        reader = file1.readlines()
        data = []
        for row in reader:                      #reads in each row
            temp1 = ""
            for t in row:
                temp1 += t
            temp2 = [int(e) if e.isdigit() else e for e in temp1.split(',')]
            stats = np.array(temp2)
            data.append(stats)
        patients = np.vstack(data)
        norm_stats = preprocessing.normalize(patients)
    return norm_stats                           #returns array

def dist(train_point, test_point):              #calculates distance between 2 data points, returns int
    sum = 0
    for i in range(len(train_point)):
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
        i = 0
        d = 0
        min = 10000
        for c in clusters:
            if not (t==c[0]).all():
                i += 1
        if i >= len(clusters):
            for r in range(len(clusters)):
                d1 = dist(t, clusters[r][0])
                if min > d1:
                    min = d1
                    d = r
            clusters[d].append(t)
    return clusters

def nearest(cluster, point):
    min = dist(cluster[0], point)
    temp = cluster[0]
    for c in cluster:
        d1 = dist(c, point)
        if d1 < min:
            temp = c
            min = d1
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
    for c in cluster:
        centers.append(c[0])
    return centers

def check(cluster, centers):
    for c in cluster:
        for c1 in centers:
            if (c1==c[0]).all():
                return False
    return True

train = "data-1.txt"
start_time = time.time()
training = norm_ary(train)
k = 2
x_axis = []
y_axis = []
while k <= 15:
    print("starting k = ",k)
    counter1 = 10
    minimum = []
    while counter1 > 0:
        clusters = []
        temp1 = []
        temp1.append(training[0])
        rando = list(random.sample(range(1,len(training)), k-1))
        print('random numbers: ', rando)
        clusters.append(temp1)
        i = 2
        for r in rando:
            temp = []
            temp.append(training[r])
            clusters.append(temp)
            i += 1
        centers = list(clusters)
        temp2 = 0
        counter = 0
        while 1:
            clusters = list(k_mean(training, clusters))
            if abs(len(clusters[0]) - temp2) < .00001:
                break
            temp2 = len(clusters[0])
            temp_clusters = []
            for r in range(k):
                temp0 = []
                temp0.append(centroid(clusters[r]))
                temp_clusters.append(temp0)
            clusters = list(temp_clusters)
            counter += 1
        counter1 -=1
        minimum.append(counter)
    print(minimum,"min: ", min(minimum))
    y_axis.append(min(minimum, key=float))
    x_axis.append(k)
    k += 1
print("--- %s seconds ---" % (time.time() - start_time))
plt.plot(x_axis, y_axis)  #plots training, testing, and LOOCV errors of various k values
plt.show()
