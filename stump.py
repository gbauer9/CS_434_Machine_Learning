#Part 2: section 1: Decision Stump
import csv
import numpy as np
import math
from sklearn import preprocessing
import operator
from collections import Counter
from numpy import genfromtxt

#*******************************
#Decided to implement the CART algorithm.(clasification and regression trees) 
#In this version, for entropy calculation, the gini index is used. 
#I did research on both of these algorithms, and both should produce 
#the same result. I made this choice simply because Gini is easier to implement
#Online resources were used for inspiration -- those are cited below.
#********************************
def main():
	
	print "Problem 2, Section 1"
	#First thing to do is gather data from .csv files
	#The general gini algorithm I followed required the diagnosis values
	#in this case, -1 and 1, to be in the last column. The next lines
	#switches the first column with the last in both training and testing data
	train = genfromtxt('knn_train.csv', delimiter=',')
	test = genfromtxt('knn_test.csv', delimiter=',')

	#swap columns:
	#all columns excluding the first
	temp = train[:,1:31]
	np_temp = np.array(temp)
	temp2 = train[:,0:1]
	np_temp2 = np.array(temp2)
	Full_train = np.append(np_temp, np_temp2, axis=1)
	
	#do the same for the testing data:
	temp = test[:,1:31]
	np_temp = np.array(temp)
	temp2 = test[:,0:1]
	np_temp2 = np.array(temp2)
	Full_test = np.append(np_temp, np_temp2, axis=1)

	#Step one: call greedy create_tree method	   
	tree = create_tree(Full_train)
	split_2(tree)
	print "Information Gain: "
	print tree["infogain"]
	print "Training CORRECT rate: "
	print stats(tree, Full_train)
	print "Training ERROR rate: "
	print ( 1 - stats(tree, Full_train))
	print "Testing CORRECT rate: "
	print stats(tree, Full_test)
	print "Testing INCORRECT rate: "
	print ( 1- stats(tree, Full_test))
	
	return 0

#****************************************
#This replaces our ENTROPY method. The gini index is used 
#as the information gain for this decision tree
#Measures how good splits are
#followed a generic gini algorithm from online
#***************************************
def entropy(groups):

	diagnosis = [-1, 1]
	n_instances = float(sum([len(group) for group in groups]))
	entropy_val = 0.0
	for group in groups:
		size = len(group)
		if size == 0:
			continue
		score = 0.0
		for cancer in diagnosis:
			p = [row[-1] for row in group].count(cancer) / float(size)
			score += p * p 
		entropy_val += (1.0 - score) * (size /n_instances)
	return entropy_val
#**********************************************
#Greedy induction algorithm to create the tree. This function
#calls split_tree() onto the data for binary splits as mentioned in 
#the assignment. It then compares the computed gini index (entropy) 
#and stops when it finds the best gini index.
#The return is a dictionary that will mapy the key data for later
#************************************************
def create_tree(data):

	index, bestValue, bestScore, bestGroups = 100, 100, 100, 100
	#loop 31 times for each attribute:
	for i in range(31):
		for row in data:
			groups = split_tree(i, row[i], data)
			entropy_val = entropy(groups)	
			if entropy_val < bestScore and entropy_val != 0:
			#kept getting caught here, had to make sure gini wasn't zero
				index, bestValue, bestScore, bestGroups =i, row[i], entropy_val ,groups
	return { "index": index, "value": bestValue, "groups": bestGroups, "infogain": bestScore}
#*********************************************
#Split_tree = 	takes in an index of the attribute to split
#		takes in a split value for the attribute.
#		iterates over each row checking if the attribute value is below 
#		or above the split value, which assigns it to the left or right group. 
#**********************************************
def split_tree(i, v, data):

	left, right = list(), list()
	for row in data:
		if row[i] < v:
			left.append(row)
		else: 
			right.append(row)
	
	return left, right	
#******************************
#Takes in the root node of the tree, should be two groups.
# first defines left and right subtrees, and stores each group into it.
# calls count_occurences() to determine whether the left or right subgroups get
# labeled as "-1", and "1" 
#********************************
def split_2(root):

	left, right = root["groups"]
	root["left"], root["right"] = count_occurences(left), count_occurences(right)
	return
#*********************************
#Chooses the highest occurency for each side of the tree, either 1 or -1.
#**********************************
def count_occurences(section):
	majority = [row[-1] for row in section]
	return max(set(majority), key=majority.count)

#***********************************
#This simply compares the prediction from our predict function
#with our created tree to see how correct we were. 
#Simply increments a correct counter every time, and then divides by the
#number for a ratio. Assignment wants error of incorrect, so we sub from 1
#**************************************
def stats(tree,data):
	correct = 0
	for row in data:
		prediction = predict(tree,row)
		if prediction == row[-1]:
			correct += 1
	return (float(correct) / len(data))

#***************************************
#makeprediction will recursively iterate through the tree to make a simple prediction
#		Takes in current node in the tree from (get error rate)
#		takes in the current row in the dataset from (get error rate)
#The basic Idea for this specific prediction function came from:
#  https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
#****************************************
def predict(treeNode, row):
	
	if row[treeNode["index"]] < treeNode["value"]:
		if isinstance(treeNode["left"], dict):
			return predict(treeNode["left"], row)
		else:
			return treeNode["left"]
	else:
		if isinstance(treeNode["right"], dict):
			return predict(treeNode["right"], row)
		else:
			return treeNode["right"]

main()
	
