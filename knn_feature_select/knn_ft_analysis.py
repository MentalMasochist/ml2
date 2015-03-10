# developer: Richard Brooks
# description: randomized optimization to find optimal portfolio combination
# contact: richardbrks@gatech.edu

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import bitflip_randOpt as ropt
import pandas as pd
import numpy as np
import datetime as dt
import copy
import matplotlib.pyplot as plt
from random import random
import chow_liu_trees as CLT
from collections import defaultdict
import networkx as nx
import csv

np.random.seed(None)

def read_data(filename, feature_cols, label_col):
	""" read supervised data.
	"""
	features = []
	labels = []
	with open(filename, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		col_names = csvreader.next()
		for line in csvreader:
			line = np.array(line)
			features.append(line[feature_cols])
			labels.append(line[label_col])
	return np.array(features), np.array(labels)

def main():
	"""
	"""
	# read data
	trnfname = './data/ContraceptiveChoice/cmc-train.csv'
	na_trnfeatures, na_trnlabels = read_data(trnfname, np.arange(9), np.array(9))
	tstfname = './data/ContraceptiveChoice/cmc-train.csv'
	na_tstfeatures, na_tstlabels = read_data(tstfname, np.arange(9), np.array(9))

	# define fitness function: 5 fold-cross validation accuracy
	def fit(vec_select):
		# last 7 digits are for k
		bin_k = vec_select[-7:].astype(str)
		k = int("".join(bin_k),2)
		if k == 0:
			return -1
		if np.sum(vec_select[:-7]) == 0:
			return -1
		# create learner
		clf = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(clf, na_trnfeatures[:,vec_select[:-7].astype(bool)], na_trnlabels, cv=5)
		return scores.mean()

	# test algorithms
	# Randomized Hill CLimbing
	print "\nRHC " + "#"*30
	ofilename = "./results/rhc_results.csv"
	max_fit, max_vec = ropt.bitflip_RandomizedHillClimb(na_trnfeatures.shape[1]+7, fit, iterations=100, restarts=1, verbose=True)

	# # Simulated Annealing
	# print "\nSA " + "#"*30
	# ofilename = "./results/sa_results.csv"
	# max_fit, max_vec = ropt.bitflip_SimulatedAnnealing(na_trnfeatures.shape[1]+7, fit, iterations=100, restarts=1, verbose=True)

	# # Genetic Algorithms
	# print "\nGA " + "#"*30
	# ofilename = "./results/ga_results.csv"
	# max_fit, max_vec = ropt.bitflip_GeneticAlgorithm(na_trnfeatures.shape[1]+7, fit, population_size=100, generations=3, restarts=1, child_per_parent=1, mutation_prob=0.1, crossover_split=0.5, verbose=True)

	# # MIMIC
	# print "\nMIMIC " + "#"*30
	# ofilename = "./results/mimic_results.csv"
	# max_fit, max_vec = \
	# 	ropt.bitflip_MIMIC(na_trnfeatures.shape[1]+7, fit, iterations=10, restarts=1, init_population_size=100, n_samples=100, theta_percentile=0.8, verbose=True)

	# get results on test data
	feature_vec = max_vec[:-7]
	max_k = int("".join(max_vec[-7:].astype(str)),2)
	clf = KNeighborsClassifier(n_neighbors=max_k)
	clf.fit(na_trnfeatures[:,feature_vec.astype(bool)], na_trnlabels)
	y_predict = clf.predict(na_tstfeatures[:,feature_vec.astype(bool)])
	test_acc = np.sum(y_predict == na_tstlabels) / float(len(na_tstlabels))
	print "test_acc = %f" % test_acc

	# write results
	writer = csv.writer(open(ofilename,'w'))
	writer.writerow(['train_acc',max_fit])
	writer.writerow(['feature_vector',feature_vec])
	writer.writerow(['k_neighbors',max_k])
	writer.writerow(['test_acc',test_acc])

	return


if __name__ == "__main__":
	main()