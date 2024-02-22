import csv
import numpy as np

dat = np.array(list(csv.reader(open("../data/ratings2015W2V2.csv", 'r'))))
dat = dat[:,[1,2,3,7]]

d = dict()
d2 = dict()
d3 = dict()
users = np.sort(np.unique(dat[1:,0]).astype(int)).astype(str)

for row in users:
	d[row] = []
	d2[row] = []
	d3[row] = []

for row in dat[1:]:
	d[row[0]].append(row[1])
	d2[row[0]].append(row[2])
	d3[row[0]].append(row[3])

out = []
out2 = []
out3 = []
for row in users:
	temp = []
	temp2 = []
	temp3 = []
	temp.append(row)
	temp2.append(row)
	temp3.append(row)
	for row2 in d[row]:
		temp.append(row2)
	for row2 in d2[row]:
		temp2.append(row2)
	for row2 in d3[row]:
		temp3.append(row2)
	out.append(temp)
	out2.append(temp2)
	out3.append(temp3)

wr = csv.writer(open("../data/users2015_seq2.csv", 'w'))
wr.writerows(out)

wr2 = csv.writer(open("../data/users2015_rating2.csv", 'w'))
wr2.writerows(out2)

wr3 = csv.writer(open("../data/users2015_topic2.csv", 'w'))
wr3.writerows(out3)
