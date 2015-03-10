"""
Description: Assignment 2 NeuralNetwork analysis
Developer: Richard Brooks
"""

import continuous_randOpt as ropt

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from sklearn import cross_validation

import numpy as np
import csv
import sys


def Evaluation():
	"""  Randomized Hill Climbing
	"""
	# inputs
	debug = False
	trnfname = "./data/cmc-train.csv"
	tstfname = "./data/cmc-test.csv"
	ofname = "./results/cmc-SA_eval_results.csv"
	# read dataset
	trndata = read_data(trnfname)
	tstdata = read_data(tstfname)
	# create neural network
	nodes = int( (trndata.indim + trndata.outdim) / 2 ) # Weka ['a']
	fnn = buildNetwork( trndata.indim, nodes, trndata.outdim, outclass=SoftmaxLayer )
	
	def fitness(params):
		for i in range(len(params)):
			fnn.params[i] = params[i]
		y_predict = fnn.activateOnDataset(trndata)
		y_predict = np.apply_along_axis(np.argmax, 1, y_predict)
		y_act = np.apply_along_axis(np.argmax, 1, trndata['target'])
		train_acc = np.sum(y_predict == y_act) / float(len(y_predict))
		return train_acc

	# Randomized Hill Climbing
	# 100 iter : 18.5 sec
	ofilename = "./results/rhc_results.csv"
	# max_fit, max_vec, d_progress = ropt.continuous_RandomizedHillClimb(len(fnn.params), fitness, iterations=10000, restarts=10, verbose=False)
	max_fit, max_vec = ropt.continuous_RandomizedHillClimb(len(fnn.params), fitness, iterations=100, restarts=1, verbose=False)

	# # Simulated Annealing
	# # # 100 iter : 23.1 sec
	# ofilename = "./results/sa_results.csv"
	# # max_fit, max_vec = ropt.continuous_SimulatedAnnealing(len(fnn.params), fitness, iterations=10000, restarts=10, verbose=False)
	# max_fit, max_vec = ropt.continuous_SimulatedAnnealing(len(fnn.params), fitness, iterations=100, restarts=1, verbose=False)

	# # Genetic Algorithms
	# # 100 iter : 42 seconds
	# ofilename = "./results/ga_results.csv"
	# # max_fit, max_vec = ropt.continuous_GeneticAlgorithm(len(fnn.params), fitness, population_size=100, generations=100, restarts=5, verbose=False)
	# max_fit, max_vec = ropt.continuous_GeneticAlgorithm(len(fnn.params), fitness, population_size=100, generations=10, restarts=1, verbose=False)

	# test randomized optimization solution
	for i,v in enumerate(max_vec):
		fnn.params[i] = v
	y_predict = fnn.activateOnDataset(tstdata)
	y_predict = np.apply_along_axis(np.argmax, 1, y_predict)
	y_act = np.apply_along_axis(np.argmax, 1, tstdata['target'])
	test_acc = np.sum(y_predict == y_act) / float(len(y_predict))

	print "train acc = %f " % max_fit
	print "test acc = %f" % test_acc
	print "weights:"
	print max_vec

	writer = csv.writer(open(ofilename, 'w'))
	writer.writerow(['train_acc',max_fit])
	writer.writerow(['test_acc',test_acc])
	writer.writerow(['weights',max_vec])

	# print fnn.params[:10]
	# trainer = BackpropTrainer(fnn, dataset=trndata, batchlearning=True) 
	# print 100.0 - percentError( trainer.testOnClassData(dataset=tstdata), tstdata['class'] )
	# epochs = 5
	# trainer.trainEpochs( epochs )
	# print fnn.params[:10]


	# # Back Propogation Evaluation
	# print fnn.params[:5]
	# epochs = 5
	# trainer = BackpropTrainer( fnn, dataset=tstdata, verbose=False )
	# trainer.trainEpochs( epochs )
	# fitness = 100.0 - percentError( trainer.testOnClassData(dataset=tstdata), tstdata['class'] )
	# print fnn.params[:5]
	# print fitness

	return



def read_data(fname):
	"""
	"""
	reader = csv.reader(open(fname,'r'))
	reader.next() # skip headers
	data = []
	for line in reader:
		data.append(line)
	data = np.array(data).astype(int)
	data = get_pybrain_data(data)
	return data

def backprop_cv():
	""" Main cross-validation program
	"""
	# inputs
	debug = True
	trnfname = "./data/cmc-train.csv"
	ofname = "./results/cmc-cv_results.csv"
	writer = csv.writer(open(ofname, 'w'))
	ls_learningrate = np.arange(0.01, 0.06, 0.01)
	ls_momentum = np.arange(0.0, 0.05125, 0.0125)
	epochs = 100
	
	# read datasets
	ls_trndata, ls_tstdata = get_folded_data(trnfname, folds=5)

	if debug:
		print "Number of training patterns: ", len(ls_trndata[0])
		print "input and output dimensions: ", ls_trndata[0].indim, ls_trndata[0].outdim
		print "First sample (input, target, class):"
		print ls_trndata[0]['input'][0], ls_trndata[0]['target'][0], ls_trndata[0]['class'][0]
		print

	# perform cross validation
	print "learningrate,momentum,epochs,cv_accuracy,insample_accuracy"
	writer.writerow("learningrate,momentum,epochs,cv_accuracy,insample_accuracy".split(","))
	for learningrate in ls_learningrate:
		for momentum in ls_momentum:
			cv_accuracy, insample_accuracy = get_backprop_cv_accuracy(ls_trndata, ls_tstdata, learningrate=learningrate, momentum=momentum, epochs=epochs)
			print "%s,%s,%s,%s,%s" % (learningrate, momentum, epochs, cv_accuracy, insample_accuracy) 
			# writer.writerow([learningrate, momentum, epochs, cv_accuracy, insample_accuracy])
	return


def get_backprop_cv_accuracy(ls_trndata, ls_tstdata, learningrate, momentum, epochs):
	""" calculate cross-validatoin accuracy
	"""
	cv_accuracy = 0 
	insample_accuracy = 0 
	for i in range(len(ls_tstdata)):
		trndata = ls_trndata[i]
		tstdata = ls_tstdata[i]
		# build learner
		nodes = int( (trndata.indim + trndata.outdim) / 2 ) # Weka ['a']
		fnn = buildNetwork( trndata.indim, nodes, trndata.outdim, outclass=SoftmaxLayer )
		# train model
		trainer = BackpropTrainer( fnn, dataset=tstdata, learningrate=learningrate, momentum=momentum, verbose=False, weightdecay=0.00 )
		trainer.trainEpochs( epochs )
		cv_accuracy += 100.0 - percentError( trainer.testOnClassData(dataset=tstdata), tstdata['class'] )
		insample_accuracy += 100.0 - percentError( trainer.testOnClassData(dataset=trndata), tstdata['class'] )
	cv_accuracy = cv_accuracy / len(ls_tstdata)
	insample_accuracy = insample_accuracy / len(ls_tstdata)
	return cv_accuracy, insample_accuracy

def get_folded_data(fname, folds=5):
	""" get data ready for cross-validation analysis
	"""
	reader = csv.reader(open(fname,'r'))
	reader.next() # skip headers
	data = []
	for line in reader:
		data.append(line)
	data = np.array(data).astype(int)
	kf = cross_validation.KFold(data.shape[0], n_folds=folds, shuffle=True, random_state=0)
	ls_kf_trndata = []
	ls_kf_tstdata = []
	for trn_index, test_index in kf:
		ls_kf_trndata.append( get_pybrain_data( data[trn_index] ) )
		ls_kf_tstdata.append( get_pybrain_data( data[test_index] ) )
	return ls_kf_trndata, ls_kf_tstdata

def get_pybrain_data(data):
	""" convert to pybrain dataset
	"""
	features = data[:,:-1] 
	labels = data[:,-1]
	inputdim = features.shape[1]
	nb_classes = len(np.unique(labels))
	DS = ClassificationDataSet(inputdim, nb_classes=nb_classes)
	for i, ft in enumerate(features):
		DS.appendLinked( ft, [labels[i]-1] ) # labels need to be a range of int starting from 0 (I know, this is stupid)
	DS._convertToOneOfMany( )
	return DS

if __name__ == "__main__":
	# backprop_cv()
	Evaluation()