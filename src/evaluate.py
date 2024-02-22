import csv
import sys
import copy
import getopt
import numpy as np
import ml_metrics as metric
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# percentile function
f_percentile = stats.percentileofscore

def find_by_query(dat, ind, query, order='o'):
	if order == 'o':
		out = []
		out2 = []
		for row in dat:
			in_flag = 1
			seq_flag = 1
			gap_flag = 1
			inds = []

			for q in query:
				if q not in row[1:]:
					in_flag = 0
					break
				inds.append(row[1:].index(q))

			if not in_flag:
				continue

			if len(query) == 1:
				out.append(row[inds[-1]+1:])
				out2.append(row[0])
				continue

			for i in range(len(inds)-1):
				if inds[i] > inds[i+1]:
					seq_flag = 0
					break
			if not seq_flag:
				continue

#			inds.sort()
			gaps = []
			for i in range(len(inds)-1):
				g = inds[i+1] - inds[i]
				if g > maxgap:
					gap_flag = 0
					break
				gaps.append(g)

			if seq_flag and gap_flag:
#				print "user "+str(i)+"\ngap:"
#				print [g_temp for g_temp in gaps]
				# for movies after query
#				out.append(row[inds[-1]+1:])	
				# for movies in user sequence except query
				temp = copy.copy(row[1:])	
				for q in query:
					temp.remove(q)
				out.append(temp)
				out2.append(row[0])

		return out, out2

	elif order == 'n':
		out = []
		out2 = []
		i = 0
		for row in dat:
			i += 1
			in_flag = 1
			seq_flag = 1
			gap_flag = 1
			inds = []

			for q in query:
				if q not in row[1:]:
					in_flag = 0
					break
				inds.append(row[1:].index(q))

			if not in_flag:
				continue

			if len(query) == 1:
				out.append(row[inds[-1]+1:])
				out2.append(row[0])
				continue
			
			inds.sort()
			gaps = []
			for i in range(len(inds)-1):
				g = inds[i+1] - inds[i]
				if g > maxgap:
					gap_flag = 0
					break
				gaps.append(g)

			if gap_flag:
#				print "user "+str(i)+"\ngap:"
#				print [g_temp for g_temp in gaps]
				# for movies after query
#				out.append(row[inds[-1]+1:])
				# for movies in user sequence except query
				temp = copy.copy(row[1:])
				for q in query:
					temp.remove(q)
				out.append(temp)
				out2.append(row[0])

		return out, out2

	else:
		return [], []

def accuracy(recommended, tests):
	aps = 0
	for row in tests:
		ap = metric.apk(row,recommended,len(recommended))
		aps += ap
	if len(tests) < 1:
		res = 0
	else:
		res = float(aps)/float(len(tests))
#		res = metric.mapk(tests, recommended, k=len(recommended))
	return res

def diversity(recommended, topics, n_topics=35):
	topic = []
	for i in recommended:
		try:
			temp = topics[np.where(topics[:,0]==i)[0][0]][1]
			topic.append(temp)
		except:
			pass
		
	topic = np.array(topic)
	n_topics_rec = np.unique(topic).shape[0]
	res = n_topics_rec / float(n_topics)
	return res

def satisfaction(recommended, test_ind, movies, ratings):
	score_sum = 0.
	n_hit_users = 0
	for user in test_ind:
		rating_sum = 0.
		hits = 0.
		rating_of_user = list(np.array(ratings[indices.index(user)][1:]).astype('float32'))
		movies_of_user = movies[indices.index(user)][1:]
		for rec in recommended:
			if rec in movies[indices.index(user)][1:]:
				rating_sum += float(rating_of_user[movies_of_user.index(rec)])
#				hit_sum += float(ratings[indices.index(user)][movies[indices.index(user)].index(rec)])
				hits += 1
		if hits == 0:
			continue
		else:
			n_hit_users += 1
		rating_rank = f_percentile(rating_of_user, (rating_sum / float(hits))) * 0.01
		hit_ratio = hits / float(len(recommended))
		score = rating_rank * hit_ratio
		score_sum += score
		
	if n_hit_users == 0:
		res = 0
	else:
		satiscore = score_sum / float(n_hit_users)
		res = satiscore

	return res

def make_patch_spines_invisible(ax):
	ax.set_frame_on(True)
	ax.patch.set_visible(False)
	for sp in ax.spines.values():
		sp.set_visible(False)

def plotting():
	fig, host = plt.subplots()
	fig.subplots_adjust(right=0.75)
	plt.title("context: "+context+" gap: "+str(maxgap)+" ordered: "+order_)

	par = host.twinx()
	par2 = host.twinx()
	par2.spines["right"].set_position(("axes", 1.2))
	make_patch_spines_invisible(par2)
	par2.spines["right"].set_visible(True)

	p1, = host.plot(range(1, len(recommended)), accuracies, "b-", label="MAP")
	p2, = par.plot(range(1, len(recommended)), diversities, "r-", label="Diversity")
	p3, = par2.plot(range(1, len(recommended)), satiscores, "g-", label="Satisfaction")

	host.set_ylim(0,1)
	par.set_ylim(0,1)
	par2.set_ylim(0,1)
	

	host.set_xlabel("# of Recommendations")
	host.set_ylabel("Mean Average Precision")
	par.set_ylabel("Diversity")
	par2.set_ylabel("Satisfaction")

	host.yaxis.label.set_color(p1.get_color())
	par.yaxis.label.set_color(p2.get_color())
	par2.yaxis.label.set_color(p3.get_color())

	tkw = dict(size=4, width=1.5)
	host.tick_params(axis='y', colors=p1.get_color(), **tkw)
	par.tick_params(axis='y', colors=p2.get_color(), **tkw)
	par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
	host.tick_params(axis='x', **tkw)

	lines = [p1, p2, p3]
	host.legend(lines, [l.get_label() for l in lines])

	plt.savefig("../results/eval_w"+str(window)+"_q"+str(n_query)+"_"+context+"_g"+str(maxgap)+"_"+order_+".png")

# main
try:
	opt, args = getopt.getopt(sys.argv[1:], "i:w:q:c:g:o:")
except:
	print str(err)
	sys.exit(1)

if not opt:
	print "usage: python evaluate.py -i inputfile -w window -q query -c context -g maxgap -o ordered"
	sys.exit(1)

default = False
for op, p in opt:
	if op == '-i':
		input_rec = p
	elif op == '-w':
		window = int(p)
	elif op == '-q':
		n_query = int(p)
	elif op == '-c':
		context = p
	elif op == '-g':
		maxgap = int(p)
	elif op == '-o':
		order_ = p
	else:
		default = True

dat = list(csv.reader(open("../data/Testset_seq.csv", 'r')))
dat2 = np.array(list(csv.reader(open(input_rec, 'r'))))
topics = np.array(list(csv.reader(open("../data/TopicVector.csv", 'r'))))
ratings = list(csv.reader(open("../data/Testset_rating.csv", 'r')))

if default:
	n_query = 3		# number of query items
	maxgap = 10
	window = 8
	context = 'r'
	order_ = 'n'

n_query_ = n_query+1	# for indexing
query = dat2[1:n_query_,0]
recommended = list(dat2[n_query_:,0])

indices = []
for i in dat:
	indices.append(i[0])

n_topics = np.unique(topics[:,1]).shape[0]

queried_testset, test_ind = find_by_query(dat, indices, query, order=order_)
print str(len(queried_testset))+" queried users found"

accuracies = []
diversities = []
satiscores = []
for	i in range(1, len(recommended)):
	if i % 50 == 0 or i == len(recommended)-1:
		print i
	recs = copy.copy(recommended[:i])
	acc = accuracy(recs, queried_testset)
	div = diversity(recs, topics)
	sat = satisfaction(recs, test_ind, dat, ratings) 
	accuracies.append(acc)
	diversities.append(div)
	satiscores.append(sat)
	 	
print 'Mean Average Precision score:',accuracies[-1]
print 'Diversity score:',diversities[-1]
print 'Satisfaction score:',satiscores[-1]

plotting()

