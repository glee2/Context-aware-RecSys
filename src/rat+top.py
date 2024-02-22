import csv
import numpy as np

ratings = np.array(list(csv.reader(open("../data/ratings2015.csv", 'r'))))
topics = np.array(list(csv.reader(open("../data/topics2015.csv", 'r'))))

toparr = np.zeros((ratings.shape[0],1)).astype('int')

for i in range(1,topics.shape[0]):
	ind = np.where(ratings[:,1]==topics[i][0])
	toparr[ind] += int(topics[i][1])

