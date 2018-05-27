import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
from numpy.linalg import eigh, solve
from numpy import array, dot, mean, std, empty, argsort
from sklearn import preprocessing
import pylab as plt

def norm_ary(file):                   #reads in file, converts to a matix, normalizes and returns the array
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
	return norm_stats

#****************************************
# Pack data into nxd matrix
#	-Rows correspond to examples, columns correspond to features
# compute d x d covariance matrix E
# Calculate the eigen vectors/values of E (using numpy software package)
# Rank eigen values in decreasing order
#	- i-th eigen value = the variance of data after projecting onto i-th eigen vector
#	- Choose the highest -> retain the most variance
#Select top k eigenvectors


#Compute the covariance matrix from the data
#Use the covariance matrix to produce the eigenvalues and eigenvectors
def cov(data):

	m , n = data.shape

	#compute mean
	data -= data.mean(axis=0)

	cov_matrix = np.cov(data, rowvar=False)
	#eigh function returns 2 arrays:
	#	the first is a 1-d array containing eigen values
	#	the second is a 2-d array of the corresponding eigenvectors (in columns)
	evals, evecs = eigh(cov_matrix)
	#sort the eigenvalues in decreasing order.
	sort = np.argsort(evals)[::-1]
	#sort the eigenvectors in the exact same way
	evecs = evecs[:, sort]
	evals = evals[sort]

	#select the first 10, as per assignment instructions
	evecs = evecs[:, :10]
	eval_top = evals[:10]

	print("10 top eigen-values: ")	
	print(eval_top)
	plt.imshow(np.reshape(evecs,28,28))



data = "data-1.txt"
x = norm_ary(data)
cov(x)
