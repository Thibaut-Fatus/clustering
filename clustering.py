# coding=utf-8
import csv, time, random, operator, sys

import numpy as np
from numpy import genfromtxt
from scipy.spatial import distance
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from sklearn.metrics.pairwise import euclidean_distances

from collections import defaultdict

import pandas as pd

## filename for classifier
filename_clf = 'dataset_1M_boostedinterest.csv'
filename_clf = 'dataset_100k_boostedinterest.csv'
filename_clf = 'dataset_100k_bin.csv'
#filename_clf = 'dataset_1M_bin.csv'

## filename for pandas stats
filename_pd = 'dataset_100k_bin.csv'

## classifier relative
choose_clf = 'kmeans' # [kmeans, dbscan]
n_clusters = 4
val = 5 ## value given for weighted distance (interests / market intent)
n_components = 5 #PCA
use_pca = True
n_points = 2000
metric=distance.euclidean
binary = True


## cosine c'est trop long .. meme sur 100k, DBSCAN sur PCA ou autre c'est bizarre .. toutes les distances manuelles ca marche pas 
## en use pca les decrits sont pas mal avec DBSCAN, mais pas la visu 
## ward en memory error, comme le precompute sur les distances ...
## dbscan sans pca en binaire prend trop de temps
## dbscan(0.3,20) marche pas mal en pca5 binaire sur 100k pareil en 0.3,100, (4/5 clusters) mais pas en 1M .. (180, 4000s de calcul..)

## mapper to get dict of dimensions and subdimensions
print '### loading mapper ... ###'
with open('map.csv', 'rb') as f:
  r = csv.reader(f)
  dim_ids = set()
  dims = dict()
  ssdims = dict()
  for row in r:
    dim = row[0]
    dim_id = int(row[1])
    ssdim = row[2] 
    ssdim_id = int(row[3])
    if dim_id not in dim_ids:
      dim_ids.add(dim_id)
      dims[dim_id] = dim
      ssdims[dim_id] = dict()
    ssdims[dim_id][ssdim_id] = ssdim
print 'ok !'

## mask to get dims / subdims from indices
print '### creating mask ...'
sorted_dims = sorted(dims.iteritems(), key=operator.itemgetter(0))
mask = dict()
mask_ssd = dict()
i = 0
always_bin = set(['Market intent', 'Interests', 'Funnel client'])
for d_id, d_name in sorted_dims:
  if binary or d_name in always_bin:
    for ssd in ssdims[d_id]:
      mask[i] = d_id
      mask_ssd[i] = ssd
      i += 1
  else:
    mask[i] = d_id
    i += 1
print "ok !"
#print mask
#print mask_ssd

## file loading for classification
print "### loading file : %s ###" % filename_clf
data = genfromtxt(filename_clf, delimiter=',')
print "data loaded ! length : %s" % len(data)

## compute PCA if necessary and project data
if (use_pca):
  print "### using PCA with %s components" % n_components
#pca = PCA(n_components=n_digits).fit(data)
  pca = PCA(n_components=n_components)
  reduced_data = pca.fit_transform(data)

## classifiers initialisation  
#mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=10000,
#                      n_init=5, max_no_improvement=10, verbose=0)
km = KMeans(init='k-means++', n_clusters=n_clusters, precompute_distances=False,  
            n_init=10,  verbose=0)
dbs100k = DBSCAN(eps=0.3, min_samples=20)#,metric=metric)
dbs1M = DBSCAN(eps=0.3, min_samples=100)#,metric=metric)
#ward = Ward(n_clusters=n_clusters)
#mbk = MiniBatchKMeans(init=pca.components_, n_clusters=3, batch_size=1000,
#                      n_init=1, max_no_improvement=10, verbose=0)

## classifier choice from imput string (need to be arg ..)
print "### classifier: %s ###" % choose_clf
if choose_clf == 'kmeans':
  clf = km
elif choose_clf == 'dbscan':
  clf = dbs
else:
  print "error: unknown classifier (%s) " % choose_clf
  sys.exit(-1)

## fitting data
print "### fitting data ... ###"
#t0 = time.time()
#mbk.fit(data)
#print 'ok : %s s' % (time.time()-t0)
t0 = time.time()
if use_pca:
  print "on projected data"
  clf.fit(reduced_data)
else:
  clf.fit(data)
print 'ok : %s s' % (time.time()-t0)
print "all good .."

## retrieving infos for display, clusters stats ...
labels = clf.labels_
got_veters = False
try:
  cluster_centers = clf.cluster_centers_
  got_centers = True
except:
  print 'cannot have centers of clusters for algo : %s' % choose_clf
labels_unique = np.unique(labels)

pop = defaultdict(int)

for l in labels:
  pop[l] += 1

n_clusters = len(labels_unique)

print "### labels: %s " % ','.join([str(int(l)) for l in labels_unique])

nb_rows = len(data[0])
filters_per_label = dict()
for l in [int(a) for a in labels_unique]:
  filters_per_label[l] = []
  for r in range(nb_rows):
    #filters_per_label[l] = [defaultdict(int)] * nb_rows #does not work : same dict !
    filters_per_label[l].append(defaultdict(int))

##count_combi = defaultdict(int)
print "### computing filters per labels ...  ###"
t0 = time.time()
for i in range(len(labels)):
##
 # z = ''.join([str(int(x)) for x in data[i]])
 # count_combi[z] += 1
##
  for e in range(nb_rows):
    filters_per_label[int(labels[i])][e][data[i][e]] += 1
print 'ok : %s s' % (time.time()-t0)

##print "combinaisons : %s " % len(count_combi)
 #sorted_count = sorted(count_combi.iteritems(), key=operator.itemgetter(1), reverse = True)
 #for i in range(100):
##  print sorted_count[i]
 

for p,nb in pop.items():
  print '###################'
  print "label %s ( %s ):" % (p, nb)
  print '###################'
  if binary:
    ldim = dict()
    bdim = dict()
    for d, _ in sorted_dims:
      ldim[d] = []
    for i in range(nb_rows):
      e = filters_per_label[p][i]
      ldim[mask[i]].append([100*(float(e[1])/(e[0]+e[1])), e[0], e[1], ssdims[mask[i]][mask_ssd[i]]])
    for d, name in sorted_dims:
      ldim[d].sort()
      ldim[d].reverse()
      try:
        bdim[d] = ldim[d][0][0]
      except:
        print 'no data, skipping'
    best_dims = sorted(bdim.iteritems(), key=operator.itemgetter(1), reverse = True)
    #print best_dims
    for d,s in best_dims:
      print dims[d]
      for ele in ldim[d][:10]:
        print "\t%s : %s in, %s out ( %.2f %%)" % (ele[3], ele[2], ele[1], ele[0])
      print "\t\t__ __ __"

  else:
    li = []
    lmi = []
    for i in range(nb_rows):
      e = filters_per_label[p][i]
      if i < 4:
        st = ', '.join(['%s : %s' % (ssdims[mask[i]][int(x)],y) for x,y in e.items()])
        print "%s : %s" % (dims[i], st)
      else:
        if i < 21:
          li.append([100*(float(e[val])/(e[0]+e[val])), e[0], e[val], ssdims[mask[i]][mask_ssd[i]]])
        else:
          lmi.append([100*(float(e[val])/(e[0]+e[val])), e[0], e[val], ssdims[mask[i]][mask_ssd[i]]])
    li.sort()
    lmi.sort()
    li.reverse()
    lmi.reverse()
    print "Interests:"
    for ele in li[:3]:
      print "\t%s : %s / %s ( %.2f %%)" % (ele[3], ele[2], ele[1], ele[0])
    print "\t\t__ __ __"
    for ele in li[-3:]:
      print "\t%s : %s / %s ( %.2f %%)" % (ele[3], ele[2], ele[1], ele[0])
    print "Market Intents:"
    for ele in lmi[:2]:
      print "\t%s : %s / %s ( %.2f %%)" % (ele[3], ele[2], ele[1], ele[0])
    print "\t\t__ __ __"
    for ele in lmi[-2:]:
      print "\t%s : %s / %s ( %.2f %%)" % (ele[3], ele[2], ele[1], ele[0])
    print "\n"

## compute if possible distance between clusters
if got_centers:
  center_distances = euclidean_distances(cluster_centers)
  print "\tlabel\tlabel\tdistance"
  for i in labels_unique:
    for j in labels_unique:
      if i < j:
        print "\t%s\t%s\t%.3f" % (i, j, center_distances[i][j])

color={-1:'c',0:'r', 1:'b', 2:'g', 3:'y', 4:'k', 5:'m', 6:'c', 7:0.1, 8:0.3, 9:0.8}
for i in xrange(10, 100):
  color[i]=random.random()

def get_color(c):
  if c < 100:
    return color[c]
  else:
    return 0.5


if (use_pca):
  print pca.explained_variance_ratio_
#  print pca.components_
if (n_components == 2 and use_pca):
  # Step size of the mesh. Decrease to increase the quality of the VQ.
  h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
  
  # Plot the decision boundary. For that, we will assign a color to each
  x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
  y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  
  # Obtain labels for each point in mesh. Use last trained model.
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  
  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  pl.figure(1)
  pl.clf()
  pl.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=pl.cm.Paired,
            aspect='auto', origin='lower')
  
  pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
  # Plot the centroids as a white X
  centroids = clf.cluster_centers_
  pl.scatter(centroids[:, 0], centroids[:, 1], 
             marker='x', s=169, linewidths=3,
             color='w', zorder=10)
  pl.title('K-means clustering(PCA-reduced data)\n'
           'Centroids are marked with white cross')
  pl.xlim(x_min, x_max)
  pl.ylim(y_min, y_max)
  pl.xticks(())
  pl.yticks(())
  pl.show()

elif (n_components >= 3 and use_pca):
  fignum = 1
  mar={0:"o", 1:"s", 2:"^", 3:"*", 4:"8", 5:"v", 6:"D"}
  for i in range(n_components):
    for j in range(i + 1, n_components):
      for k in range(j + 1, n_components):

        fig = pl.figure(fignum, figsize=(4, 3))
        pl.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        pl.cla()
        

        #ax.scatter(data[:n_points, i], data[:n_points, j], data[:n_points, k], c=labels[:n_points].astype(np.float))
        ax.scatter(reduced_data[:n_points, i], reduced_data[:n_points, j], reduced_data[:n_points, k], c=[get_color(int(l)) for l in labels[:n_points]]) #, marker=[mar[l] for l in labels[:n_points]])
        ax.scatter(reduced_data[-n_points:, i], reduced_data[-n_points:, j], reduced_data[-n_points:, k], c=[get_color(int(l)) for l in labels[-n_points:]]) #, marker=[mar[l] for l in labels[:n_points]])
        #ax.plot_wireframe(data[:n_points, i], data[:n_points, j], data[:n_points, k])#, extend3d=True, c=labels[:n_points].astype(np.float))

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('PCA%s' % i)
        ax.set_ylabel('PCA%s' % j)
        ax.set_zlabel('PCA%s' % k)
        fignum += 1


pl.show() 
