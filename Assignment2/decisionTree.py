# Much of the code for building the tree came from here, modified to use entropy instead of Gini index: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
import numpy as np
import math

CLASS_VALUES = [-1.0, 1.0]

# Split the dataset using given feature/value
def try_split(feature, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[feature] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
# Calculate the entropy for a given split
def calc_uncertainty(groups):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	
	total_uncertainty = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		entropy = 0.0
		# Calculate the entropy for each group
		for class_val in CLASS_VALUES:
			p = [row[0] for row in group].count(class_val) / size
			if p == 0:
				entropy = 0.0
				continue
			else:
				entropy += -p * math.log2(p)
		# Sum up the total uncertainty, weighting it by group size
		total_uncertainty += entropy * (size / n_instances)
	return total_uncertainty
 
# Select the best split point for a dataset
def find_best_split(dataset):
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(1, len(dataset[0])):
		for row in dataset:
			groups = try_split(index, row[index], dataset)
			uncertainty = calc_uncertainty(groups)
			if uncertainty < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], uncertainty, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def create_terminal(group):
	outcomes = [row[0] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = create_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = create_terminal(left), create_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = create_terminal(left)
	else:
		node['left'] = find_best_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = create_terminal(right)
	else:
		node['right'] = find_best_split(right)
		split(node['right'], max_depth, min_size, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = find_best_split(train)
	split(root, max_depth, min_size, 1)
	return root

# Make a prediction with a decision tree via recursion
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']


# Automate testing a tree using given dataset, return the error rate (# of failures / total # of datapoints)
def test_predictions(dataset, tree):
	failure_count = 0.0
	
	for row in dataset:
		prediction = predict(tree, row)
		if prediction != row[0]:
			failure_count += 1.0

	return failure_count / float(len(dataset)) 

def main():
	training_data = np.loadtxt("knn_train.csv", delimiter=',')
	testing_data = np.loadtxt("knn_test.csv", delimiter=',')


	print("Training Error Rate\tTesting Error Rate\tDepth")
	for depth in range(1, 7):
		tree = build_tree(training_data, depth, 1)

		error_rate_train = test_predictions(training_data, tree)
		error_rate_test = test_predictions(testing_data, tree)

		print(str(error_rate_train) + '\t' + str(error_rate_test) + '\t' + str(depth))
	

if __name__ == "__main__":
	main()