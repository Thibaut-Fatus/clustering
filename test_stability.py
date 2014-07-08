# -*- coding: utf-8 -*-

import numpy as np
from numpy import genfromtxt
import math, copy

from sklearn.decomposition import PCA

from sklearn import metrics, preprocessing
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN

from user_mask import *

from collections import defaultdict

from itertools import permutations

filename_list = []
for i in range(1,11):
#for i in range(1,3):
  filename_list.append("test_validity/100k_t%s.csv" %i)

#for i in range(1,9):
#for i in range(1,6):
#  filename_list.append("test_validity/1M_t%s.csv" %i)

n_components = 5
n_clusters = 8

clf = dict()

print "### using PCA with %s components ###" % n_components

components = dict()
ids = dict()
reduced_data = dict()
labels = dict()

## file loading for classification              
def getPCA(f, i):
  print "### loading file : %s ###" % f
  data = genfromtxt(f, delimiter=',')
  print "data loaded ! length : %s" % len(data)    
  data_ini = data #keep data structure for display
  
  pca = PCA(n_components=n_components)
  reduced_data[i] = pca.fit_transform(data)
  components[i] = pca.components_
  return pca
  
## compute mean and max dot 1 vs 1 dot product for each components of pcas (created over different datasets)
def computeMeanDot(nb_test):
  c = 0
  mean_dot = defaultdict(float)
  max_dot = defaultdict(float)
  for i in range(nb_test):
    for j in range(i + 1, nb_test):
      c += 1
      for compo in range(n_components):
        v = abs(1 - abs(np.dot(components[i][compo], components[j][compo])))
        mean_dot[compo] += v
        if v > max_dot[compo]:
          max_dot[compo] = v
  for k,v in mean_dot.items():
    mean_dot[k] = v / c
  return mean_dot, max_dot

def computeClassifier(nb_test):
  global reduced_data, clf, labels
  clusters_centers = dict()
  for i in range(nb_test):
    clf[i] = KMeans(init='k-means++', n_clusters=n_clusters, precompute_distances=False,
                    n_init=10,  verbose=0, n_jobs=3, tol=1e-5)
    labels[i] = clf[i].fit_predict(reduced_data[i])
    print "computed classifier %s " % i
  return clf

## euclidian distance
def distance(x,y):
  d = 0
  #for i in range(n_components):
  #  d += (x[i] - y[i]) **2
  #return math.sqrt(d)
  return np.dot(x,y)/(math.sqrt(np.dot(x,x))*math.sqrt(np.dot(y,y)))

## sum of distance between cluster centers c1 and c2 for a given permutation p
def localDistance(c1, c2, p):
  i = 0
  d = 0
  while (i < n_clusters):
    d += distance(c1[i], c2[p[i]])
    i += 1
  return d
  
## tries to find best pairs of clusters from 2 lists of centers
def findBestCombo(clf1, clf2):
  c1 = clf1.cluster_centers_
  c2 = clf2.cluster_centers_
  p = permutations(range(8), 8)
  run = True
  tested = 0
  min_dist = 1000000
  min_dist = -1000000 ## if COSINE
  best_perm = []
  while run:
    try:
      current = p.next()
      local_dist = localDistance(c1, c2, current)
      #if (local_dist < min_dist):
      if (local_dist > min_dist): ## if COSINE
        min_dist = local_dist
        best_perm = copy.deepcopy(current)
      tested += 1
      #if (tested % 10000 == 0):
      #  print "checked %s ..." % tested
    except:
      run = False
      return min_dist, best_perm
        
## returns n_points per cluster (used for mean distance to center)
def getPoints(n_points, cl_id):
  global labels, reduced_data
  lab = labels[cl_id]
  red_data = reduced_data[cl_id]
  needed = n_points * n_clusters
  stored = defaultdict(int)
  res = dict()
  for i in range(n_clusters):
    res[i] = []
  i = 0
  while needed > 0 and i < len(lab):
    if stored[lab[i]] < n_points:
      stored[lab[i]] += 1
      needed = needed - 1
      res[lab[i]].append(red_data[i])
    i += 1
  return res

## compute mean distance between subset of points of a cluster and this cluster center
def getMeanDistToCenter(points, cl_id):
  centers = clf[cl_id].cluster_centers_
  nb_points = len(points[0])
  dist = dict()
  for i in range(len(centers)):
    center = centers[i]
    d = 0
    for p in points[i]:
      d += distance(p, center)
    dist[i] = float(d) / nb_points
  #print 'mean distance to center'
  #print dist
  mean_dist = 0
  for k, v in dist.items():
    mean_dist += v
  return (mean_dist / n_clusters)

## compute distances between centers (as a matrix)
def computeCentersDist(c):
  dist = dict()
  centers = c.cluster_centers_
  for i in range(n_clusters):
    dist[i] = dict()
  mean_dist = 0
  count = 0
  for i in range(n_clusters):
    for j in range(i + 1, n_clusters):
      dist[i][j] = distance(centers[i], centers[j])
      dist[j][i] = dist[i][j]
      mean_dist += dist[i][j]
      count += 1
  #print "distance between centers"
  #print dist
  return (mean_dist / count)



def main():
  i = 0
  for f in filename_list:
    ids[f] = i
    getPCA(f, i)
    i += 1
  nb_test = i ## nb files used for testing + 1
  mean_d, max_d = computeMeanDot(nb_test)
  print "## dot products between PC of PCA ##"
  for k,v in mean_d.items():
    print "%s : %s ( max : %s )" % (k,v, max_d[k])
  
  clf = computeClassifier(nb_test)

  for i in range(nb_test):
    print "#### test %s ####" % i
    print "mean distance between centers"
    print computeCentersDist(clf[i])
    test_points = getPoints(1000, i)
    print "mean distance to centers"
    print getMeanDistToCenter(test_points, i)
    print "####         ####\n"
  
  mean_min_dist = 0
  count = 0
  for i in range(nb_test):
    for j in range(i + 1, nb_test):
      min_d, per = findBestCombo(clf[i], clf[j])
      mean_min_dist += (min_d / n_clusters)
      count += 1
      print "i, j : %s, %s" % (i, j)
      print "optimal mean distance between clusters centers: %s " % str(float(min_d) / n_clusters)
      print "best permutation : %s " % str(per)
  print "####"
  print "mean minimum distance between clusters centers"
  print str(mean_min_dist / count)
  


if __name__ == "__main__" :
  main()
