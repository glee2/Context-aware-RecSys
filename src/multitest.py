from multiprocessing import Process as worker
import multiprocessing
import time

def cal(n):
	suma = 0
	for i in range(n):
		suma += i
	return suma
s = time.time()
pool = multiprocessing.Pool(processes=1)
print pool.map(cal, [50000000])
pool.close()
pool.join()

print "%s" % (time.time() - s)
