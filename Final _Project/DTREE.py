#Decision Tree
import csv
import numpy as np
import math
import time
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import operator
from collections import Counter
import pylab as plt
from datetime import*
from decimal import*
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
    return data                         #returns array




train = "Subject_4.csv"
training = norm_ary(train)
X = training
y = []
for entry in X:
	val = entry[-1]
	y.append(int(val))

clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)


test = "Subject_1.csv"
testing = norm_ary(test)
z = []
for line in testing:
	val = line[-1]
	z.append(int(val))
	

y_pred = clf.predict(testing)

print ("Accuracy: ", accuracy_score(z, y_pred, normalize = False))





