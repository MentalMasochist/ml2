# developer: Richard Brooks
# description: randomized optimization to find optimal portfolio combination
# contact: richardbrks@gatech.edu

import color_randOpt as ropt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
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
	nodes = []
	edges = []
	with open(filename, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=' ')
		col_names = csvreader.next()
		for line in csvreader:
			line = np.array(line).astype(int)
			nodes.append(line[0])
			nodes.append(line[1])
			edges.append((line[0],line[1]))
	return np.unique(np.array(nodes)), edges

def main():
	"""
	"""
	# read data
	# filename = './data/gc_1000_5'
	filename = './data/gc_4_1'
	na_nodes, ls_edges = read_data(filename, np.arange(9), np.array(9))

	# create graph from the edges
	G = nx.Graph()
	G.add_edges_from(ls_edges)

	# define fitness function: 5 fold-cross validation accuracy
	def fit(vec_color):
		ct_match = 0
		max_color = np.max(vec_color) 
		min_color = np.min(vec_color) 
		for e in G.edges():
			if vec_color[e[0]] != vec_color[e[1]]:
				ct_match += 1
		fitness = ct_match*10**int(np.ceil(np.log10(2*len(vec_color)))) - max_color - min_color
		return fitness

	# test algorithms
	# Randomized Hill CLimbing
	print "\nRHC " + "#"*30
	ofilename = "./results/rhc_results.csv"
	max_fit, max_vec = ropt.color_RandomizedHillClimb(len(G.nodes()), fit, iterations=100, restarts=1, verbose=True)

	# # Simulated Annealing
	# print "\nSA " + "#"*30
	# ofilename = "./results/sa_results.csv"
	# max_fit, max_vec = ropt.color_SimulatedAnnealing(len(G.nodes()), fit, iterations=1000, restarts=1, verbose=True)

	# # Genetic Algorithms
	# print "\nGA " + "#"*30
	# ofilename = "./results/ga_results.csv"
	# max_fit, max_vec = ropt.color_GeneticAlgorithm(len(G.nodes()), fit, population_size=100, generations=10, restarts=1, child_per_parent=1, mutation_prob=0.1, crossover_split=0.5, verbose=True)
	
	# # MIMIC
	# print "\nMIMIC " + "#"*30
	# ofilename = "./results/mimic_results.csv"
	# max_fit, max_vec = \
	# 	ropt.color_MIMIC(len(G.nodes()), fit, iterations=10, restarts=1, init_population_size=100, n_samples=100, theta_percentile=0.8, verbose=True)

	# write results
	writer = csv.writer(open(ofilename,'w'))
	writer.writerow(['fitness',max_fit])
	writer.writerow(['selection_vector',max_vec])
	writer.writerow(['colors',len(np.unique(max_vec))])

	return


if __name__ == "__main__":
	main()