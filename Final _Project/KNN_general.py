#KNN
import csv
import numpy as np
import math
from sklearn import preprocessing
import operator
from collections import Counter
import pylab as plt
from decimal import *
from datetime import*
import os

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
    getcontext().prec = 10
    window_timeframe = 7
    with open(file) as file1 :
        reader = file1.readlines()
        data = []
        window = []
        for row in reader:                      #reads in each row
            temp1 = [e for e in row.split(',')]
            temp_date = list([Decimal(datetime.strptime(temp1[0], "%Y-%m-%dT%H:%M:%SZ").hour)])
            temp2 = temp_date + [Decimal(t) for t in temp1[1:]]
            if len(window) < window_timeframe:
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
        if len(window) < window_timeframe and len(window) > 0:                     #appends any unfinished data points
            if temp2[-1] <= 0:
                stats = make_wind(window)
                data.append(stats)
        patients = np.vstack(data)
        norm_stats = preprocessing.normalize(patients)
        for n in norm_stats:
            if n[-1] > 0:
                n[-1] = 1
    return norm_stats                           #returns array

def norm_test(file):                             #reads in file, converts to a matix, normalizes and returns the array
    getcontext().prec = 3
    window_timeframe = 7
    with open(file) as file1 :
        reader = file1.readlines()
        data = []
        i = 0
        sum = 0
        for row in reader:
            window = []
            temp00 = [e for e in row.split(',')]
            for r in temp00:
                if i == 6:
                    if r.isdigit():
                        r = Decimal(r)
                    sum += Decimal(r)
                    window.append(sum/7)
                    sum = 0
                    i = 0
                else:
                    if r.isdigit():
                        r = Decimal(r)
                    sum += Decimal(r)
                    i += 1
            window = np.array(window)
            data.append(window)
        patients = np.vstack(data)
        norm_stats = preprocessing.normalize(patients)
    return norm_stats                           #returns array

def dist(train_point, test_point):              #calculates distance between 2 data points, returns int
    sum = 0
    for i in range(len(train_point)-1):
        sum += pow(train_point[i] - test_point[i],2)
    return math.sqrt(sum)                       #returns int

def nearest_k(train_set, new_point, k, i = None):       #finds nearest k neighbors, has them vote and returns the winning vote
    neighbors = []
    if i is None:                                       #when not using leave one out cross validation
        for t in train_set:
            delta_x = dist(t, new_point)
            neighbors.append((t, delta_x))
        neighbors.sort(key = operator.itemgetter(1))    #sorts neigbors into order accourding to distance
        nearest = []
        for r in range(k):
            #print(neighbors[r][0][-1])
            nearest.append(neighbors[r][0][-1])
            #print(Counter(nearest).most_common(1)[0][0])
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
    testing_accuracy_pos = 0
    testing_accuracy_neg = 0
    testing_accuracy_pos_err = 0
    testing_accuracy_neg_err = 0
    for t in testing:
        if t[-1] == 1:
            if t[-1] == nearest_k(training, t, k):
                testing_accuracy_pos += 1
            else:
                testing_accuracy_pos_err += 1
        elif t[-1] == 0:
            if t[-1] == nearest_k(training, t, k):
                testing_accuracy_neg += 1
            else:
                testing_accuracy_neg_err += 1
        else:
            print("crap")
    testing_accuracy = [testing_accuracy_pos, testing_accuracy_pos_err, testing_accuracy_neg, testing_accuracy_neg_err]
    return testing_accuracy                         #returns float


def training_check(train):
    training_error = []
    training_error_pos = []
    testing_error = []
    x_axis = []
    temp_test = []
    for k in range(1,14,2):                                                 #for testing and training error at various k values
        print(k)
        temp_training_error = -1
        temp_training_error_pos = -1
        temp_testing_error = -1
        for i in range(len(train)):
            temp0 = []
            temp00 = []
            for t in train:
                if t is not train[i]:
                    temp0.append(norm_ary(t))
                else:
                    temp00.append(norm_ary(t))
            training = np.vstack(temp0)
            training1 = np.vstack(temp00)
            #testing = list(norm_ary(test))
            acc = accuracy(training, training1, k)
            if temp_training_error_pos < acc[0]/(acc[0]+acc[1]):
                temp_training_error_pos = acc[0]/(acc[0]+acc[1])
            if temp_training_error < ((acc[0]/(acc[0]+acc[1]))+(acc[2]/(acc[2]+acc[3])))/2:
                temp_training_error = ((acc[0]/(acc[0]+acc[1]))+(acc[2]/(acc[2]+acc[3])))/2
        training_error.append(temp_training_error)
        training_error_pos.append(temp_training_error_pos)
        x_axis.append(k)
    plt.plot(x_axis, training_error, 'r--', x_axis, training_error_pos, 'b--')  #plots training, testing, and LOOCV errors of various k values
    plt.show()                                                                  #shows above plot


train = ["Subject_1.csv", "Subject_4.csv", "Subject_6.csv", "Subject_9.csv"]
#training_check(train)
k = 3                   #k=3 had the best predictive results
final_train = ["Subject_1.csv", "Subject_4.csv", "Subject_6.csv"]       # leaveing our Subject_9 has the best impact on predictions
results = []
temp0 = []
for f in final_train:
    temp0.append(norm_ary(f))
final_training = np.vstack(temp0)
print("done with training setup")
test = "general_test_instances.csv"
testing = norm_test(test)

print("done with testing setup")
for t in testing:
    results.append(int(nearest_k(final_training, t, k)))
with open('general_pred1.csv', 'w', newline='') as file1:
    write = csv.writer(file1)
    write.writerows(results)
