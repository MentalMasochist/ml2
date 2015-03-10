# developer: Richard Brooks
# description: randomized optimization for continuous parameters
# contact: richardbrks@gatech.edu

import numpy as np
import datetime as dt
import copy
from random import random

np.random.seed(None)

def get_param_neighbor(param):
	""" switch one bit.
	"""
	new_param = copy.deepcopy(param)            # copy vector
	bit_idx = np.random.randint(len(new_param))  # get a random bit
	new_param[bit_idx] += random()-0.5           # move [-0.5,0.5]
	return new_param


def continuous_RandomizedHillClimb(param_size, fit_func, iterations, restarts, verbose=False):
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
		param = np.random.random(param_size)*2-1
		# start with initial values
		old_fit = fit_func(param)	
		old_vec = param
		# initialize max values (on first restart)
		if max_fit == None:
			max_fit = old_fit
			max_vec = old_vec
		for i in xrange(iterations):
			# take a random walk
			new_param = get_param_neighbor(param) 
			# get current fit
			new_fit = fit_func(new_param)
			# assign a new fit function
			if new_fit >= max_fit:
				max_fit = new_fit
				max_vec = new_param
				param = copy.deepcopy(new_param)
				if verbose:
					print "max fit: %10.6f" % max_fit
			iter_ct += 1
			if (iter_ct % record_ct == 0):
				d_progress[iter_ct] = max_fit
	return max_fit, max_vec, d_progress


def continuous_SimulatedAnnealing(param_size, fit_func, iterations, restarts, T=1.0, alpha=0.999, verbose=False):
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
		param = np.random.random(param_size)*2-1
		# start with initial values
		old_fit = fit_func(param)	
		old_vec = param
		# initialize max values (on first restart)
		if max_fit == None:
			max_fit = old_fit
			max_vec = old_vec
		for i in xrange(iterations):
			# sample a neighbor
			new_vec = get_param_neighbor(param) 		
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


def continuous_GeneticAlgorithm(param_size, fit_func, population_size=100, generations=100, restarts=1, child_per_parent=1, mutation_prob=0.1, crossover_split=0.5, verbose=False):
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
		population = np.random.random((population_size, param_size))*2-1
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
		offspring = list(get_param_neighbor(np.array(offspring)))
	return offspring


def main():
	return

if __name__ == "__main__":
	main()