## this script compute pca transformation and then labelisation of all elements of input file. 
## could be use directly with scanresp to assign a cluster to each element of our db
## WARNIG remind that our clusters excluded some elements regarding proportion (90% female -> remove 10% male ..)

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
import pickle
import csv, time


filehandler = open("pca.pk","rb")

pca = pickle.load(filehandler)

filehandler = open("clf.pk","rb")

clf = pickle.load(filehandler)

def fromCsv():
  i = 1
  with open('dump_from_db_1M.csv','rb') as f:
    t0 = time.time() 
    for r in f:
      print r
      clf.predict(pca.transform(r))
      i += 1
      if i % 1000 == 0:
        t1 = time.time()
        print 1000/(t1-t0)
        t0 = t1

def getLabel(x):
  return clf.predict(pca.transform(x))
