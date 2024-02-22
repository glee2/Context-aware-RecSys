import csv
import sys
import copy
import getopt
import time
import numpy as np
import ml_metrics as metric
from scipy import stats
from scipy.spatial import distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# percentile function
f_percentile = stats.percentileofscore

def evaluating(user_ind):
	seq = np.array(user_d[user_ind]['seq'])
	top = np.array(user_d[user_ind]['top'])
	rat = np.array(user_d[user_ind]['rat'])

	rand_start = np.random.randint(len(seq) - 20)
	train_ind = range(rand_start, rand_start+10)
	train_seq = seq[train_ind]

#	temp_seq = copy.copy(train_seq)
#	for i in range(len(train_seq)):
#		if train_seq[i] not in item_indices:
#			temp_seq = np.delete(temp_seq, np.where(temp_seq==train_seq[i]))
#	train_seq = copy.copy(temp_seq)

	recommended = recommend(train_seq)
	test_ind = range(rand_start+10, rand_start+20)
	test_seq = seq[test_ind]

#	temp_seq = copy.copy(test_seq)
#	for i in range(len(test_seq)):
#		if test_seq[i] not in item_indices:
#			temp_seq = np.delete(temp_seq, np.where(temp_seq==test_seq[i]))
#	test_seq = copy.copy(temp_seq)
	
#	if len(test_seq) == 0:
#		return -1, -1

	res_acc = len(list(set(test_seq).intersection(recommended))) / float(len(test_seq))
	res_div = diversity(recommended, test_seq)

	return res_acc, res_div

#def hit_ratio(test, recommended):
	

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

	res = sim_vecs[sim_vecs[:,1].astype('float32').argsort()[-200:][::-1]][:,0]
		
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
print "done\n",time.time() - start," elapsed"

item_indices = np.sort(wordvecs[1:].astype(float), axis=0)[:,0].astype(int).astype(str)
user_indices = []
for user in seqs:
	user_indices.append(user[0])

n_topics = 6

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
start = time.time()
acc_sum = 0.
div_sum = 0.
acc_sum_cf = 0.
div_sum_cf = 0.
n_tested = 0.
accs, divs, accs_cf, divs_cf = [], [], [], []
user_sample = np.random.randint(len(user_indices)-20, size=500)
n_rep = 10

# user index is different for W2V and CF --> need to be equal
for u in user_sample:
	u_acc = 0.
	u_div = 0.
	u_acc_cf = 0.
	u_div_cf = 0.

	test_ind = np.where(cf_test[u]>0)[0]
	test_len = test_ind.shape[0]

	if test_len == 0 or len(seqs[u][1:]) < 21:
		continue

	for _ in range(n_rep):
		# W2V 
		temp_acc, temp_div = evaluating(user_indices[u])
		u_acc += temp_acc
		u_div += temp_div

	avg_u_acc = u_acc / n_rep
	avg_u_div = u_div / n_rep
	acc_sum += avg_u_acc
	div_sum += avg_u_div
	accs.append(avg_u_acc)
	divs.append(avg_u_div)
	
	# CF
	rec = item_indices[np.argsort(cf_pred[u])[::-1]][:test_len]
	test = item_indices[test_ind]
	temp_acc_cf = len(list(set(test).intersection(rec))) / float(test_len)
	temp_div_cf = diversity(rec, test)
	acc_sum_cf += temp_acc_cf
	div_sum_cf += temp_div_cf
	accs_cf.append(temp_acc_cf)
	divs_cf.append(temp_div_cf)

	n_tested += 1.

acc = acc_sum / n_tested
div = div_sum / n_tested
acc_cf = acc_sum_cf / n_tested
div_cf = div_sum_cf / n_tested

print "evaluation for",n_tested,"users done\n",time.time() - start," elapsed"
print "W2V result"
print "Hit ratio :",acc
print "Diversity :",div
print "CF result"
print "Hit ratio :",acc_cf
print "Diversity :",div_cf

plot_eval(accs, divs, accs_cf, divs_cf)

