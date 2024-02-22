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

def evaluating(user_ind, type = 'w2v', n_train = 10, n_test = 10, n_rec_ = 200):
	uid = user_indices[user_ind]
	seq = np.array(user_d[uid]['seq'])
	top = np.array(user_d[uid]['top'])
	rat = np.array(user_d[uid]['rat']).astype(float)

	rand_start = np.random.randint(len(seq) - (n_train+n_test))
	train_ind = range(rand_start, rand_start+n_train)
	train_seq = seq[train_ind]
	train_rat = rat[train_ind]

	test_ind = range(rand_start+n_train, rand_start+n_train+n_test)
	test_seq = seq[test_ind]

	if type=='w2v':
		rec = recommend(train_seq, n_rec_ = n_rec_)
		res_acc = len(list(set(test_seq).intersection(rec))) / float(n_test)
		res_div = diversity(rec, test_seq)
	
		return res_acc, res_div

	elif type=='w2v+d':
		if user_cluster[user_ind][1] == 'Stayer':

			rec1 = recommend(train_seq)
			acc_w2v = len(list(set(test_seq).intersection(rec1))) / float(n_test)
			div_w2v = diversity(rec1, test_seq, n_rec_ = n_rec_)

			temp_jumpers = []
			for i in user_indices[np.where(user_cluster[:,1]=='Jumper')[0]]:
				temp_jumpers += user_d[i]['seq']
			temp_jumpers = np.unique(temp_jumpers)
			temp = np.isin(topics_dic[:,0], temp_jumpers)
			current_topic = topics_dic[:,1][np.where(topics_dic[:,0]==train_seq[len(train_seq)-n_replace-1])[0]]
			other_topics = topics_dic[temp][np.where(topics_dic[temp][:,1]==current_topic)[0]][:,0]
			other_topics_ = np.setdiff1d(other_topics, np.hstack((train_seq,test_seq)))
			new = np.random.choice(other_topics_, n_replace, replace=False)
			train_seq = np.hstack((np.delete(train_seq, range(len(train_seq)-n_replace, len(train_seq))), new))

		elif user_cluster[user_ind][1] == 'Jumper':
			other_topics = item_indices[np.where(topics_dic[:,1]==topics_dic[:,1][np.where(topics_dic[:,0]==train_seq[len(train_seq)-n_replace-1])[0]])[0]]
			other_topics_ = np.setdiff1d(other_topics, np.hstack((train_seq,test_seq)))
			new = np.random.choice(other_topics_, n_replace, replace=False)
			train_seq = np.hstack((np.delete(train_seq, range(len(train_seq)-n_replace, len(train_seq))), new))

		else:
			pass
		
		rec = recommend(train_seq, n_rec_ = n_rec_)
		
		res_acc = len(list(set(test_seq).intersection(rec))) / float(n_test)
		res_div = diversity(rec, test_seq)

		if user_cluster[user_ind][1] == 'Stayer':
			if acc_w2v < res_acc:
				return res_acc, res_div
			else:
				for i in other_topics_:
					train_seq = np.hstack((np.delete(train_seq, range(len(train_seq)-n_replace, len(train_seq))), i))
					rec = recommend(train_seq, n_rec_ = n_rec_)
					res_acc = len(list(set(test_seq).intersection(rec))) / float(n_test)
					res_div = diversity(rec, test_seq)
					if acc_w2v < res_acc:
						return res_acc, res_div
						break

		return res_acc, res_div

	elif type=='cf':
		temp_cf_mat = cf_mat[user_ind]
		cf_mat[user_ind] = np.zeros(cf_mat[user_ind].shape)
		cf_mat[user_ind][np.where(np.isin(item_indices, train_seq)==True)[0]] = train_rat
		user_sim = pairdis(cf_mat, metric='cosine')
		cf_pred = predict_cf(cf_mat, user_sim)
		rec = item_indices[np.argsort(cf_pred[user_ind])[::-1]][:n_rec_]
		res_acc = len(list(set(test_seq).intersection(rec))) / float(n_test)
		res_div = diversity(rec, test_seq)
		cf_mat[user_ind] = temp_cf_mat

		return res_acc, res_div

	elif type=='rd':
		rand_ind = np.random.choice(n_items_total, n_rec_, replace=False)
		rec = item_indices[rand_ind]
		res_acc = len(list(set(test_seq).intersection(rec))) / float(n_test)
		res_div = diversity(rec, test_seq)

		return res_acc, res_div
	
def recommend(query_seq, n_rec_ = 200):
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

	res = sim_vecs[sim_vecs[:,1].astype('float32').argsort()[-n_rec_:][::-1]][:,0]
		
	return res

def diversity(recommended, test):
	topics_rec = []
	for movie in recommended:
		topics_rec.append(topics_dic[:,1][np.where(topics_dic[:,0]==movie)[0]][0])
	topics_rec = np.unique(topics_rec)
	topics_test = []
	for movie in test:
		topics_test.append(topics_dic[:,1][np.where(topics_dic[:,0]==movie)[0]][0])
	topics_test = np.unique(topics_test)
	
	res = (len(topics_rec) - len(list(set(topics_test).intersection(topics_rec)))) / float(n_topics)
	return res

def plot_eval(accs, divs, u_tested):
	recs = range(1,n_rec+1)

	fig = plt.figure()
	plt.title("Accuracy vs Number of recommendations")
	line = copy.copy(do_eval_for)
	markers = ['b-', 'r-', 'g-', 'y-']
	for i in range(len(do_eval_for)):
		line[i], = plt.plot(recs, accs_[do_eval_for[i]], markers[i], ms=2, label=do_eval_for[i].upper())
	plt.ylabel('Accuracy')
	plt.xlabel('Number of recommendations')
	plt.ylim(-0.1,1.1)
	plt.legend([l for l in line], [name.upper() for name in do_eval_for])
	fig.savefig("../results/eval_plot_acc_rec_u"+str(u_tested)+"_cv"+str(n_rep)+"_test"+str(n_test)+"_rec"+str(n_rec)+"_"+t_cluster+".png")
	plt.gcf().clear()

	fig2 = plt.figure()
	plt.title("Diversity vs Number of recommendations")
	line = copy.copy(do_eval_for)
	markers = ['b-', 'r-', 's-', 'y-']
	for i in range(len(do_eval_for)):
		line[i], = plt.plot(recs, divs_[do_eval_for[i]], markers[i], ms=2, label=do_eval_for[i].upper())
	plt.ylabel('Diversity')
	plt.xlabel('Number of recommendations')
	plt.ylim(-0.1,1.1)
	plt.legend([l for l in line], [name.upper() for name in do_eval_for])
	fig2.savefig("../results/eval_plot_div_rec_u"+str(u_tested)+"_cv"+str(n_rep)+"_test"+str(n_test)+"_rec"+str(n_rec)+"_"+t_cluster+".png")
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
topics_dic = np.array(list(csv.reader(open("../data/Topics_6.csv", 'r'))))
user_cluster = np.array(list(csv.reader(open("../data/users2015_clusters.csv", 'r'))))

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
n_rec = 200

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


print "preprocessing done\n",time.time() - start," elapsed"

print "evaluating..."

accs = { 'w2v' : [], 'w2v+d' : [], 'cf' : [], 'rd' : [] }
divs = { 'w2v' : [], 'w2v+d' : [], 'cf' : [], 'rd' : [] }
accs_ = { 'w2v' : [], 'w2v+d' : [], 'cf' : [], 'rd' : [] }
divs_ = { 'w2v' : [], 'w2v+d' : [], 'cf' : [], 'rd' : [] }
avg_u_acc = {}
avg_u_div = {}
n_user = 1
n_rep = 5
n_train = 10
n_test = 10
t_cluster = 'All'
n_replace = 1

if t_cluster == 'All':
	user_train = []
	for u in range(len(user_indices)):
		if len(seqs[u][1:]) >= (n_train+n_test+1):
			user_train.append(u)

else:
	cl_user_indices = np.where(user_cluster[:,1]==t_cluster)[0]

	user_train = []
	for u in cl_user_indices:
		if len(seqs[u][1:]) >= (n_train+n_test+1):
			user_train.append(u)

user_sample = np.random.choice(user_train, n_user, replace=False)

#do_eval_for = ['w2v', 'w2v+d', 'cf', 'rd']
do_eval_for = ['w2v', 'cf', 'rd']
#do_eval_for = ['w2v', 'w2v+d']
#do_eval_for = ['w2v+d']

if 'cf' in do_eval_for:
	cf_mat = gen_mat(df)

for method in do_eval_for:	
	for i_rec in progressbar.progressbar(range(n_rec)):
		accs = { 'w2v' : [], 'w2v+d' : [], 'cf' : [], 'rd' : [] }
		divs = { 'w2v' : [], 'w2v+d' : [], 'cf' : [], 'rd' : [] }
		start = time.time()
		u_tested = 0
		for u in user_sample:
			u_acc = 0.
			u_div = 0.
		
			for _ in range(n_rep):
				temp_acc, temp_div = evaluating(u, type = method, n_rec_ = i_rec)
				u_acc += temp_acc
				u_div += temp_div
		
			avg_u_acc[method] = u_acc / n_rep
			avg_u_div[method] = u_div / n_rep
			accs[method].append(avg_u_acc[method])
			divs[method].append(avg_u_div[method])
		
			u_tested += 1

		accs_[method].append(np.mean(accs[method]))
		divs_[method].append(np.mean(divs[method]))
		

for i in do_eval_for:
	print i.upper(),"result\n","Hit ratio :",np.mean(accs_[i]),"\nDiversity :",np.mean(divs_[i])

plot_eval(accs_, divs_, u_tested)

