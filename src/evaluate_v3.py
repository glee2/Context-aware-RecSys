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

def evaluating(user_ind):
	seq = np.array(user_d[user_ind]['seq'])
	top = np.array(user_d[user_ind]['top'])
	rat = np.array(user_d[user_ind]['rat'])

	rand_start = np.random.randint(len(seq) - 20)
	train_ind = range(rand_start, rand_start+10)
	train_seq = seq[train_ind]

	recommended = recommend(train_seq)
	test_ind = range(rand_start+10, rand_start+20)
	test_seq = seq[test_ind]

	res_acc = len(list(set(test_seq).intersection(recommended))) / float(len(test_seq))
	res_div = diversity(recommended, test_seq)

	return res_acc, res_div

def recommend(query_seq):
	query_vec = np.zeros(wordvecs_d[query_seq[0]].shape)
	movies = copy.copy(item_indices)
	for q in query_seq:
		query_vec += wordvecs_d[q]
		movies = np.delete(movies, np.where(movies==q)[0][0])

	sims = np.zeros(movies.shape[0])
	for i in range(movies.shape[0]):
		temp = 1 - distance.cosine(query_vec, wordvecs_d[movies[i]])
		sims[i] += temp
	
	sim_vecs = np.hstack((movies.reshape(-1,1), sims.reshape(-1,1)))

	res = sim_vecs[sim_vecs[:,1].astype('float32').argsort()[-n_rec:][::-1]][:,0]
		
	return res

def diversity(recommended, test):
	topics_rec = []
	for movie in recommended:
		topics_rec.append(topic_d[movie])
	topics_rec = np.unique(topics_rec)
	topics_test = []
	for movie in test:
		topics_test.append(topic_d[movie])
	topics_test = np.unique(topics_test)
	
	res = (len(topics_rec) - len(list(set(topics_test).intersection(topics_rec)))) / float(n_topics)
	return res

def plot_eval(acc_w, div_w, acc_c, div_c):
	fig = plt.figure()
	plt.title("Accuracy vs Diversity")
	line1, = plt.plot(acc_w, div_w, 'bo', ms=2, label='W2V')
	line2, = plt.plot(acc_c, div_c, 'r^', ms=2, label='UBCF')
	plt.ylabel('Diversity')
	plt.xlabel('Accuracy')
	plt.xlim(-0.1,1.1)
	plt.ylim(-0.1,1.1)
	plt.legend([line1, line2], ['W2V', 'UBCF'])
	fig.savefig('../results/evaluation.png')
	plt.gcf().clear()

def predict_cf(ratings, similarity):
	mean_user_rating = ratings.mean(axis=1)
	ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
	pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
	return pred

def gen_mat(data):
	cf_mat = np.zeros((n_users_total, n_items_total))

	for line in data.itertuples():
		cf_mat[np.where(user_indices==str(line[1]))[0][0], np.where(item_indices==str(line[2]))[0][0]] = line[3]

	return cf_mat

# main
print "data loading..."
start = time.time()
seqs = list(csv.reader(open("../data/users2015_seq2.csv", 'r')))
topics = list(csv.reader(open("../data/users2015_topic2.csv", 'r')))
ratings = list(csv.reader(open("../data/users2015_rating2.csv", 'r')))
wordvecs = np.array(list(csv.reader(open("../data/WordVectors2015.csv", 'r'))))
cf_pred = np.array(list(csv.reader(open("../data/UBCF_pred.csv", 'r')))).astype(float)
cf_test = np.array(list(csv.reader(open("../data/UBCF_testset.csv", 'r')))).astype(float)
topics_dic = np.array(list(csv.reader(open("../data/Topics_6.csv", 'r'))))
header = ['user_id','item_id','rating']
df = pd.read_csv("../data/cf_2015.csv", names=header)
n_users_total = df.user_id.unique().shape[0]
n_items_total = df.item_id.unique().shape[0]
print "done\n",time.time() - start," elapsed"

item_indices = np.sort(wordvecs[1:].astype(float), axis=0)[:,0].astype(int).astype(str)
user_indices = []
for user in seqs:
	user_indices.append(user[0])
user_indices = np.array(user_indices)

n_topics = 6
n_rec = 100

print "preprocessing..."
start = time.time()
wordvecs_d = dict()
for vec in wordvecs[1:]:
	wordvecs_d[vec[0]] = vec[1:].astype('float32')

topic_d = dict()
for t in topics_dic:
	topic_d[t[0]] = t[1]

user_d = dict()
for i in range(len(seqs)):
	user_d[seqs[i][0]] = { 'seq' : seqs[i][1:], 'rat' : ratings[i][1:], 'top' : topics[i][1:] }

print "w2v preprocessing done\n",time.time() - start," elapsed"

print "evaluating..."
acc_sum = 0.
div_sum = 0.
acc_sum_cf = 0.
div_sum_cf = 0.
n_tested = 0
n_tested_cf = 0
accs, divs, accs_cf, divs_cf = [], [], [], []
n_user = 5
user_sample = np.random.choice(len(user_indices), n_user, replace=False)
n_rep = 1
n_rep_cf = 1

# W2V
start = time.time()
for u in progressbar.progressbar(user_sample):
	u_acc = 0.
	u_div = 0.

	if len(seqs[u][1:]) < 21:
		continue

	for _ in range(n_rep):
		temp_acc, temp_div = evaluating(user_indices[u])
		u_acc += temp_acc
		u_div += temp_div

	avg_u_acc = u_acc / n_rep
	avg_u_div = u_div / n_rep
	acc_sum += avg_u_acc
	div_sum += avg_u_div
	accs.append(avg_u_acc)
	divs.append(avg_u_div)
	n_tested += 1

print "W2V for CV ",n_rep,", user ",n_tested," done\n",time.time() - start," elapsed"

# CF
start = time.time()
cf_mat = gen_mat(df)
for u in progressbar.progressbar(user_sample):
	u_acc_cf = 0.
	u_div_cf = 0.
	
	for i in range(n_rep_cf):
		rating_ind = np.where(cf_mat[u]>0)[0]
		test_len = int(0.25 * len(rating_ind))
		test_ind = np.random.choice(rating_ind, test_len, replace=False)

		cf_mat[u][test_ind] *= 0
	
		if test_len == 0:
			continue

		user_sim = pairdis(cf_mat, metric='cosine')
		cf_pred = predict_cf(cf_mat, user_sim)
		rec = item_indices[np.argsort(cf_pred[u])[::-1]][:n_rec]
		test = item_indices[test_ind]
		temp_acc_cf = len(list(set(test).intersection(rec))) / float(test_len)
		temp_div_cf = diversity(rec, test)
		u_acc_cf += temp_acc_cf
		u_div_cf += temp_div_cf

	avg_u_acc_cf = u_acc_cf / n_rep_cf
	avg_u_div_cf = u_div_cf / n_rep_cf
	acc_sum_cf += avg_u_acc_cf
	div_sum_cf += avg_u_div_cf
	accs_cf.append(avg_u_acc_cf)
	divs_cf.append(avg_u_div_cf)

	n_tested_cf += 1

print "UBCF for CV ",n_rep_cf,", user ",n_tested_cf," done\n",time.time() - start," elapsed"

acc = acc_sum / n_tested
div = div_sum / n_tested
acc_cf = acc_sum_cf / n_tested
div_cf = div_sum_cf / n_tested

#print "evaluation for",n_tested,"users done\n",time.time() - start," elapsed"
print "W2V result"
print "Hit ratio :",acc
print "Diversity :",div
print "CF result"
print "Hit ratio :",acc_cf
print "Diversity :",div_cf

plot_eval(accs, divs, accs_cf, divs_cf)

