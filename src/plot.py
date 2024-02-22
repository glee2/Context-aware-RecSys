# code simplified version of v6_user_cluster
import csv
import sys
import copy
import getopt
import time
import progressbar
import numpy as np
import pandas as pd
import ml_metrics as metric
from scipy import stats
from scipy.spatial import distance
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances as pairdis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_eval(accs, divs, u_tested):
	fig = plt.figure(figsize=(15,12), dpi=300)
	plt.title("Accuracy vs Diversity", fontsize=30, fontweight='bold')

	line = copy.copy(test_methods)
	markers = ['bo', 'r^', 'gs', 'yx']
	for i in range(len(test_methods)):
		line[i], = plt.plot(accs[test_methods[i]], divs[test_methods[i]], markers[i], ms=7, label=test_methods[i].upper())
	
	plt.ylabel('Diversity', fontsize=20)
	plt.xlabel('Accuracy', fontsize=20)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.xlim(-0.1,1.1)
	plt.ylim(-0.1,1.1)
	plt.legend([l for l in line], [name.upper() for name in test_methods], fontsize=20, prop={'size':20})
	fig.savefig("../results/[revise]_eval_plot_u300_cv20_test10_rec200_All.png")
	plt.gcf().clear()

dat = np.array(list(csv.reader(open("../results/result_v1_201806/eval_collection_u300_cv20_test10_rec200_All.csv", 'r'))))

test_methods = ['w2v', 'cf', 'rd']
accs = {'w2v' : [], 'cf' : [], 'rd' : []}
divs = {'w2v' : [], 'cf' : [], 'rd' : []}

for i in dat[:,:3].T:
	for j in i[1:]:
		accs[i[0].split('_')[-1]].append(j)

for i in dat[:,3:].T:
	for j in i[1:]:
		divs[i[0].split('_')[-1]].append(j)

for i in test_methods:
	accs[i] = np.array(accs[i]).astype('float32')
	divs[i] = np.array(divs[i]).astype('float32')

plot_eval(accs, divs, 300)
