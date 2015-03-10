# developer: Richard Brooks
# description: randomized optimization to find optimal portfolio combination
# contact: richardbrks@gatech.edu

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

def get_bitvec_neighbor(bitvec):
	""" switch one bit.
	"""
	new_bitvec = copy.deepcopy(bitvec)            # copy vector
	bit_idx = np.random.randint(len(new_bitvec))  # get a random bit
	if (new_bitvec[bit_idx] == 0):                # flip the bit 
		new_bitvec[bit_idx] = 1
	else:
		new_bitvec[bit_idx] = 0
	return new_bitvec


def bitflip_RandomizedHillClimb(bitvec_size, fit_func, iterations, restarts, verbose=False):
	""" Randomized Hill Climbing w. random restart
	"""
	# initialize max values
	max_fit = None
	max_vec = None
	iter_ct = 0
	d_progress = {}
	record_ct = 100
	for r in xrange(restarts):
		if verbose:
			print "RANDOM START"
		# create a random bit vector 
		bitvec = np.random.randint(2,size=bitvec_size)
		# start with initial values
		old_fit = fit_func(bitvec)	
		old_vec = bitvec
		# initialize max values (on first restart)
		if max_fit == None:
			max_fit = old_fit
		else:
			pass
		if max_fit == None:
			max_vec = old_vec
		else:
			pass
		for i in xrange(iterations):
			# take a random walk
			new_bitvec = copy.deepcopy(bitvec)              # copy vector
			bit_idx = np.random.randint(len(new_bitvec))	# get a random bit
			if (new_bitvec[bit_idx] == 0):                  # flip the bit 
				new_bitvec[bit_idx] = 1
			else:
				new_bitvec[bit_idx] = 0 
			# get current fit
			new_fit = fit_func(new_bitvec)
			# assign a new fit function
			if new_fit >= max_fit:
				max_fit = new_fit
				max_vec = new_bitvec
				bitvec = copy.deepcopy(new_bitvec)
				if verbose:
					print "max fit: %10.6f" % max_fit
			iter_ct += 1
			if (iter_ct % record_ct == 0):
				d_progress[iter_ct] = max_fit
	return max_fit, max_vec, d_progress


def bitflip_SimulatedAnnealing(bitvec_size, fit_func, iterations, restarts, T=1.0, alpha=0.999, verbose=False):
	""" Simulated Annealing w. random restart
		source: http://www.mit.edu/~dbertsim/papers/Optimization/Simulated%20annealing.pdf
	"""
	# initialize max values
	max_fit = None
	max_vec = None
	iter_ct = 0
	d_progress = {}
	record_ct = 100
	for r in xrange(restarts):
		if verbose:
			print "RANDOM START"
		# create a random bit vector 
		bitvec = np.random.randint(2,size=bitvec_size)
		# start with initial values
		old_fit = fit_func(bitvec)	
		old_vec = bitvec
		# initialize max values (on first restart)
		if max_fit == None:
			max_fit = old_fit
		else:
			pass
		if max_fit == None:
			max_vec = old_vec
		else:
			pass
		for i in xrange(iterations):
			# sample a neighbor
			new_vec = get_bitvec_neighbor(bitvec) 		
			# get current fit
			new_fit = fit_func(new_vec)
			# accept new vector string given acceptance probability
			acceptance_prod = np.exp(1)**(-1/T*max(0,old_fit-new_fit))
			# assign new with probability
			if acceptance_prod > random():
				old_vec = new_vec
				old_fit = new_fit
			# keep max_values
			if old_fit >= max_fit:
				max_fit = old_fit
				max_vec = old_vec
				if verbose:
					print "max fit: %10.6f" % max_fit
			if i % 100 == 0:
				T = T*alpha
			iter_ct += 1
			if (iter_ct % record_ct == 0):
				d_progress[iter_ct] = max_fit
	return max_fit, max_vec, d_progress


def bitflip_GeneticAlgorithm(bitvec_size, fit_func, population_size=100, generations=10000, restarts=1, child_per_parent=1, mutation_prob=0.1, crossover_split=0.5, verbose=False):
	""" Genetic Algorithm
		parameters:
			- population size
			- mutation probability
			- children per parent
			- population selection: only the best!
		algorithm:
			- randomly initialize each population
			- for each generation:
				- fitness is evaluated for each 
				- the more fit individuals are stochastically selected
				- genomes are modified via mutation and crossover
		source: http://en.wikipedia.org/wiki/Genetic_algorithm
	"""
	max_fitness = -np.inf 
	max_vec = None
	iter_ct = 0
	d_progress = {}
	record_ct = 100
	# random restarts
	for r in xrange(restarts):
		if verbose:
			print "RANDOM START"
		# population initialization
		population = np.random.randint(2,size=(population_size,bitvec_size))
		# for each generation
		for g in xrange(generations):
			# get the fitness for each function
			fitness = np.apply_along_axis(fit_func, 1, population)
			# select the best
			idx = np.argsort(fitness)[::-1]
			population = population[idx[:population_size]]
			# evolve and procreate
			ls_offspring = []
			for i in xrange(child_per_parent*population_size):
				ls_offspring.append(generate_offspring(population, mutation_prob, crossover_split))
			population = np.vstack((population,np.array(ls_offspring)))
			if (fitness[idx[0]] >= max_fitness):
				max_fitness = fitness[idx[0]]
				max_vec = population[idx[0],:]
				if verbose:
					print "max fit: ", fitness[idx[0]]
			for i in range(len(fitness)):
				iter_ct += 1
				if (iter_ct % record_ct == 0):
					d_progress[iter_ct] = max_fitness
	return max_fitness, max_vec, d_progress

def generate_offspring(population, mutation_prob, crossover_split):
	""" applies crossover and mutation to create 
	    new vectors for genetic algorithms
	"""
	# get parents
	parent1_id = np.random.randint(population.shape[0])
	parent2_id = np.random.randint(population.shape[0])
	parent1 = list(population[parent1_id,:])
	parent2 = list(population[parent2_id,:])
	# cross over
	split_idx = int(len(parent1)*crossover_split)
	offspring = parent1[:split_idx] + parent2[split_idx:]
	# mutate
	if mutation_prob > random():
		offspring = list(get_bitvec_neighbor(np.array(offspring)))
	return offspring

def bitflip_MIMIC(bitvec_size, fit_func, iterations=10, restarts=1, init_population_size=100, n_samples=100, theta_percentile=0.8, verbose=False):
	""" MIMIC algorithm
		source: http://www.cc.gatech.edu/~isbell/tutorials/mimic-tutorial.pdf
	"""
	max_fitness = -np.inf 
	max_vec = None
	iter_ct = 0
	d_progress = {}
	record_ct = 100
	# random restarts
	for r in xrange(restarts):
		if verbose:
			print "RANDOM START"
		# generate a random population of candidates choosen uniformly from the input space
		population = np.random.randint(2,size=(init_population_size, bitvec_size))
		# get the fitness for each function
		fitness = np.apply_along_axis(fit_func, 1, population)
		# denote the median fitness as theta(0)
		theta = np.median(fitness)
		population = population[fitness > theta,:]
		for i in xrange(iterations):
			# 1. update the parameters of the density estimator p(theta(i),x)
			# 2. generate samples from the distribution
			samples = sampleCLT(population, n_samples=n_samples)
			samples = samples.astype(int)
			population = np.vstack((population,samples))
			# 3. update theta(t+1) to only include the N-th percentile
			fitness = np.apply_along_axis(fit_func, 1, population)
			idx = np.argsort(fitness)[::-1]
			fitness = fitness[idx]
			theta_lim = int(len(fitness)*(1.0-theta_percentile))
			population = population[idx[:theta_lim],:]
			if (fitness[0] >= max_fitness):
				max_fitness = fitness[0]
				max_vec = population[0]
				if verbose == True:
					print "max value: ", max_fitness
			for i in range(len(fitness)):
				iter_ct += 1
				if (iter_ct % record_ct == 0):
					d_progress[iter_ct] = max_fitness
	return max_fitness, max_vec, d_progress

def sampleCLT(population, n_samples):
	"""
	draw a sample from a Chow-Liu tree
	"""
	samples = np.zeros((n_samples,population.shape[1]))
	X = np.apply_along_axis("".join, 1, population.astype(str))
	n = len(X[0])
	T = CLT.build_chow_liu_tree(X, n)
	for i in xrange(n_samples):
		d_sample = sampleTree(X,T,root=0)
		for j in xrange(population.shape[1]):
			samples[i,j] = d_sample[j]
	return samples

def sampleTree(X,T,root,parent=None,parent_val=None):
	"""
	sample from a chow-liu tree
	"""
	d_sample = {}
	if (parent == None): 
		d_sample[root] = sample_marginal(X,root)
		if d_sample[root] == None:
			print "sample_marginal"
			exit()
	else:
		d_sample[root] = sample_conditional(X,root,parent,parent_val)
		if d_sample[root] == None:
			print "sample_conditional"
			exit()
	for c in T.neighbors(root):
		if c == parent:
			continue
		d_sample.update(sampleTree(X,T,c,root,d_sample[root]))
	return d_sample

def sample_marginal(X, idx):
	"""
	samples from X[idx]
	"""
	ptable = CLT.marginal_distribution(X, idx)
	rand = random()
	psum = 0.
	for k,v in ptable.items():
		psum += v
		if rand <= psum:
			return k

def sample_conditional(X,node,parent,parent_cond):
	"""
	calculates pr( X[node] | X[parent]==parent_cond )
	"""
	# convert pair marginal into conditional
	pair_marginal_ptable = CLT.marginal_pair_distribution(X, parent, node)
	conditional_ptable = defaultdict(float)
	marg = 0.
	# dealing with the fact that CLT.marginal_pair_distribution will always put the smallest node first
	if parent < node:
		for (p,n),v in pair_marginal_ptable.items():
			if p == parent_cond:
				conditional_ptable[n] = v
				marg += v
	else:
		for (n,p),v in pair_marginal_ptable.items():
			if p == parent_cond:
				conditional_ptable[n] = v
				marg += v		
	for k,v in conditional_ptable.items():
		conditional_ptable[k] = v/float(marg)
	# sample from the conditional table
	rand = random()
	psum = 0.
	for k,v in conditional_ptable.items():
		psum += v
		if rand <= psum:
			return k

def main():
	return

if __name__ == "__main__":
	main()