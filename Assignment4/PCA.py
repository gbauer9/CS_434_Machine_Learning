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
	#find mean image and plot it
	mean_image = data.mean(axis=0)
	plt.imshow(np.reshape(mean_image, (28, 28)))
	plt.savefig("images/meanImage.png")

	#compute mean-normalized matrix
	data_mean_normal = data - data.mean(axis=0)

	#compute covariance matrix of mean-normalized data
	cov_matrix = np.cov(data_mean_normal, rowvar=False)

	#eigh function returns 2 arrays:
	#	the first is a 1-d array containing eigen values
	#	the second is a 2-d array of the corresponding eigenvectors (in columns)
	evals, evecs = eigh(cov_matrix)

	#find indices of evals array that would sort it in decreasing order.
	sort = np.argsort(evals)[::-1]

	#sort both the evals and evecs arrays by those indices
	evecs = evecs[:, sort]
	evals = evals[sort]

	#select the first 10, as per assignment instructions. Take the transpose of evecs to make it easier to iterate through
	evecs = evecs[:, :10].T
	eval_top = evals[:10]

	#print top 10 eigen-values
	print("10 top eigen-values: " + str(eval_top))

	#plot the top 10 eigenvectors
	for i in range(10):
		plt.imshow(np.reshape(evecs[i], (28, 28)))
		plt.savefig("images/eigenVector" + str(i) + ".png")

	#Apply the dimension reduction to each image and find the images with the highest values for each dimension
	maxVals = np.full(10, -2147483648, dtype=float)
	maxIndices = np.full(10, 0, dtype=int)
	for i in range(6000):
		#apply the dimension reduction to each datapoint
		z = np.matmul(evecs, data[i])
		for j in range(10):
			#store max value of each dimension and the index of the image 
			if z[j] > maxVals[j]:
				maxVals[j] = z[j]
				maxIndices[j] = i

	#plot the images with highest values for each dimension
	for i in range(10):
		plt.imshow(np.reshape(data[maxIndices[i]], (28, 28)))
		plt.savefig("images/maxImage" + str(i) + ".png")

data = "data-1.txt"
x = norm_ary(data)
cov(x)
