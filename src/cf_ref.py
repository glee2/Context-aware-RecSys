import numpy as np
import pandas as pd
import csv

def predict(ratings, similarity, type='user'):
	if type == 'user':
		mean_user_rating = ratings.mean(axis=1)
		ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
		pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
	elif type == 'item':
		pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
	return pred

header = ['user_id','item_id','rating']
df = pd.read_csv('../data/cf_2015.csv', names=header)
wordvecs = np.array(list(csv.reader(open("../data/WordVectors2015.csv", 'r'))))
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

user_ind = np.unique(df.user_id)
item_ind = np.unique(df.item_id)

from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances as pairdis
train, test = cv.train_test_split(df, test_size=0.25)

tr_mat = np.zeros((n_users, n_items))
te_mat = np.zeros((n_users, n_items))

for line in train.itertuples():
	tr_mat[np.where(user_ind==line[1])[0][0], np.where(item_ind==line[2])[0][0]] = line[3]
for line in test.itertuples():
	te_mat[np.where(user_ind==line[1])[0][0], np.where(item_ind==line[2])[0][0]] = line[3]

user_sim = pairdis(tr_mat, metric='cosine')
#item_sim = pairdis(tr_mat.T, metric='cosine')

#item_pred = predict(tr_mat, item_sim, type='item')
user_pred = predict(tr_mat, user_sim, type='user')

for i in range(len(user_ind)):
	test_ind = np.where(te_mat[i]>0)[0]
	test_len = test_ind.shape[0]
	if test_len == 0:
		continue
	rec = item_ind[np.argsort(user_pred[i])[::-1]][:test_len]
	test = item_ind[test_ind]
	acc = len(list(set(test).intersection(rec))) / float(test_len)
	print acc
