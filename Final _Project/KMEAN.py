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
from datetime import*
from decimal import *

def make_wind(window):      #average
    temp = []
    for r in range(len(window[0])-1):
        sum = 0
        for w in window:
            sum += w[r]
        temp.append(sum/len(window))
    temp.append(window[-1][-1])
    stats = np.array(temp)
    return stats

def norm_ary(file):                             #reads in file, converts to a matix, normalizes and returns the array
    getcontext().prec = 28
    with open(file) as file1 :
        reader = file1.readlines()
        data = []
        window = []
        for row in reader:                      #reads in each row
            temp1 = [e for e in row.split(',')]
            temp_date = list([Decimal(datetime.strptime(temp1[0], "%Y-%m-%dT%H:%M:%SZ").hour)])
            temp2 = temp_date + [Decimal(t) for t in temp1[1:]]
            if len(window) < 7:
                window.append(temp2)
                #print(window[-1])
                #print(window[-1][-1])
                if window[-1][-1] >= 1:
                    stats = make_wind(window)
                    data.append(stats)
                    window = []
            else:
                stats = make_wind(window)
                data.append(stats)
                window = []
                #stats = np.array(temp2)
        if len(window) < 7:                     #appends any unfinished data points
            if temp2[-1] <= 0:
                stats = make_wind(window)
                data.append(stats)
        patients = np.vstack(data)
        norm_stats = preprocessing.normalize(patients)
    return norm_stats                           #returns array

def dist(train_point, test_point):              #calculates distance between 2 data points, returns int
    sum = 0
    for i in range(len(train_point)):
        sum += pow(train_point[i] - test_point[i],2)
    return math.sqrt(sum)                       #returns int

def k_mean(training, clusters):     #makes clusters with list of center points
    for t in training:
        d = 0
        min = 1000000
        for r in range(len(clusters)):
            d1 = dist(t, clusters[r][0])
            if min > d1:
                min = d1
                d = r
        clusters[d].append(t)
    return clusters

def nearest(cluster, point):        #calculates the nearest point in a cluster
    min = dist(cluster[0], point)
    temp = cluster[0]
    for c in cluster:
        d = dist(c, point)
        if d < min:
            temp = c
            min = d
    return temp

def SSE(cluster, center):           #calculates SSE for a cluster
    sum = 0
    for c in cluster:
        e = pow(dist(c, center),2)
        sum += e
    sum = sum/len(cluster)
    return sum

def centroid(cluster):              #calculates center of gravity and SSE or that cluster
    temp = []
    for i in range(len(cluster[0])):
        sum = 0
        for c in cluster:
            sum += c[i]
        temp.append(sum/len(cluster))
    np.array(temp)
    temp0 = SSE(cluster, temp)
    temp = list([temp,temp0])
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

def ending(clusters, last):
    if len(last) < 1:
        #print("length less than 1")
        return False
    i = 0
    for c in clusters:
        if abs(len(c) - last[i]) < .00001:
            #print("length the same for cluster ", i)
            #print("cluster length: ", len(c), "previous length: ", last[i])
            i += 1
        else:
            #print("length not the same for cluster ", i)
            #print("cluster length: ", len(c), "previous length: ", last[i])
            return False
    return True

train = "Subject_4.csv"
start_time = datetime.now()
training = norm_ary(train)

k = 2                       #sets initial k value
x_axis = []                 # for plotting k-values
y_axis = []                 # for plotting minimum SSE values
while k <= 10:              #sets final k value
    print("starting k = ",k)
    counter1 = 50           #sets how many times each k value runs
    minimum = []
    y1_axis = []            # for plotting SSE values as it converges
    while counter1 > 0:     #runs until counter drops to 0
        clusters = []
        temp1 = []
        temp1.append(training[0])
        rando = list(random.sample(range(1,len(training)), k-1))
        clusters.append(temp1)
        i = 2
        for r in rando:     #random numbers seeding initial clusters
            temp = []
            temp.append(training[r])
            clusters.append(temp)
            i += 1
        centers = list(clusters)
        temp2 = []
        counter = 0
        while 1:                                            #runs until break statement
            clusters = list(k_mean(training, clusters))
            #print("cluster sizes:")
            #for c in clusters:
                #print(len(c))
            #print("---------------------------")
            if ending(clusters, temp2):      #break statement: when clusters stop changing size
                break
            temp2 = []
            for c in clusters:
                temp2.append(len(c))
            temp_clusters = []
            temp01 = []
            for r in range(k):
                temp0 = []
                temp_cent = list(centroid(clusters[r]))
                temp0.append(temp_cent[0])
                temp01.append(temp_cent[1])
                temp_clusters.append(temp0)
            clusters = list(temp_clusters)
            y1_axis.append(sum(temp01))
            counter += 1
        #x1_axis = list(range(1,len(y1_axis)+1))
        #plt.plot(x1_axis, y1_axis)  #plots             #plots SSE values as a function of iterations
        #plt.show()
        counter1 -=1
        minimum.append(temp01[len(temp01)-1])
    #print(minimum,"min: ", min(minimum))
    y_axis.append(min(minimum, key=float))
    x_axis.append(k)
    k += 1
plt.plot(x_axis, y_axis)                                #plots SSE values as a function of k-values
plt.show()
print("--- %s seconds ---" % (datetime.now() - start_time))
