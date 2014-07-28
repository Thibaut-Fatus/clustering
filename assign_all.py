## this script compute pca transformation and then labelisation of all elements of input file. 
## could be use directly with scanresp to assign a cluster to each element of our db
## WARNIG remind that our clusters excluded some elements regarding proportion (90% female -> remove 10% male ..)

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
import pickle
import csv, time


filehandler = open("lr_pca.pk","rb")

lr_pca = pickle.load(filehandler)

filehandler = open("lr_clf.pk","rb")

lr_clf = pickle.load(filehandler)


#filehandler = open("pca.pk","rb")
#
#pca = pickle.load(filehandler)
#
#filehandler = open("clf.pk","rb")
#
#clf = pickle.load(filehandler)


def fromCsv():
  i = 1
  with open('dump_from_db_alldim_withlabel.csv','rb') as f:
    with open('labels_lr.csv','wb') as lr:
      t0 = time.time() 
      reader = csv.reader(f)
      for r in reader:
        r_i = [map(int,x) for x in r]
        ##already done !label = clf.predict(pca.transform(r)) + 1
        label_lr = lr_clf.predict(lr_pca.transform(r_i)) + 11
        #gen.write(label)
        #gen.write('\n')
        lr.write(label_lr)
        lr.write('\n')
        i += 1
        if i % 10000 == 0:
          t1 = time.time()
          print 10000/(t1-t0)
          t0 = t1

def getLabel(x):
  return lr_clf.predict(lr_pca.transform(x))
