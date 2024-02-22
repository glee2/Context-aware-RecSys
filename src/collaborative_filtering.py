import csv
import copy
import numpy as np
from scipy.spatial import distance

dat = np.array(list(csv.reader(open("../data/ratings2015_topic6.csv", 'r'))))
seqs = list(csv.reader(open("../data/users2015_seq2.csv", 'r')))
ratings = list(csv.reader(open("../data/users2015_rating2.csv", 'r')))
wordvecs = np.array(list(csv.reader(open("../data/WordVectors2015.csv", 'r'))))[1:,0]

user_item = np.zeros((len(seqs), wordvecs.shape[0]))
item_item = np.zeros((wordvecs.shape[0], wordvecs.shape[0]))

'''
for i in range(len(seqs)):
	for j in range(len(seqs[i][1:])):
		try:
			user_item[i][np.where(wordvecs==seqs[i][1:][j])[0][0]] += float(ratings[i][1:][j])
		except:
			pass
'''


