# coding=utf-8
import csv, time, random, operator, sys, copy

import numpy as np
from numpy import genfromtxt
from scipy.spatial import distance
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from sklearn import metrics, preprocessing
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from sklearn.metrics.pairwise import euclidean_distances

from collections import defaultdict

import pandas as pd

import pickle

## local imports, not the best way to do it but needed for clarity
from config_clustering import *
from user_mask import *

## loads dimensions : data sent to front need to be octopus compliant
#from app.config.schema import dimensions_lst as dimensions

## PCA va 10* plus vite que le scale + learn, pour resultats similaires voire mieux
## cosine c'est trop long .. meme sur 100k, DBSCAN sur PCA ou autre c'est bizarre .. toutes les distances manuelles ca marche pas 
## en use pca les decrits sont pas mal avec DBSCAN, mais pas la visu 
## ward en memory error, comme le precompute sur les distances ...
## dbscan sans pca en binaire prend trop de temps
## dbscan(0.3,20) marche pas mal en pca5 binaire sur 100k pareil en 0.3,100, (4/5 clusters) mais pas en 1M .. (180, 4000s de calcul..)

## file loading for classification
print "### loading file : %s ###" % filename_clf
data = genfromtxt(filename_clf, delimiter=',')
print "data loaded ! length : %s" % len(data)
data_ini = data #keep data structure for display

if (binary and compute_global_means):
  print "### writing global stats"
  store_mean = defaultdict(int)
  total = 0
  for d in data:
    total += 1
    i = 0
    for e in d:
      store_mean[i] += e
      i += 1
  print "total : %s" % total 
  store_pn = defaultdict(str)
  for i, l in store_mean.items():
    if store_pn[dims[mask[i]]] == "":
      store_pn[dims[mask[i]]] = defaultdict(str)
    store_pn[dims[mask[i]]][ssdims[mask[i]][mask_ssd[i]]] = l
  with open(get_filename("mean_all.json"), 'wb') as f:
    f.write('{\n"count":%s' % total)
    for k, l in store_pn.items():
      ## here lies a hack for front integration :)
      if k == "Interests":
        k = 'interest'
      if k == "Market intent":
        k = 'mi'
      f.write(',\n"%s":{' % k)
      first = True
      for local_k, val in l.items():
        if first == False:
          f.write(',')
        first = False
        f.write('\n"%s":%s' % (local_k, val))
      f.write('}')
    f.write('\n}')
  print "ok!"

## reverse dimensions (18- -> (age, 0))
#reverse_dims = dict()
#for k in dimensions:
#  for e in k['bins']:
#    reverse_dims[e['name']] = (k['key'], e['key'])

## scale data if asked
if scale_data:
  data = scale(data)

## compute PCA or manual projection if necessary and project data
if (projection_mode == 1):
  print "### using PCA with %s components ###" % n_components
#pca = PCA(n_components=n_digits).fit(data)
  pca = PCA(n_components=n_components)
  reduced_data = pca.fit_transform(data)
  print pca.get_params()
  with open(get_filename("pca.pk"),"wb") as filehandler:
    pickle.dump(pca, filehandler)
  print "ok !"
elif (projection_mode >= 2):
  print "### using manual projection ###"
  n_components = 3 # for display
  # get axis
  # assign axis to mask
  nb_rows = len(mask)
  axis = [0] * nb_rows
  weight = [1] * nb_rows
  for i in range(nb_rows):
    var = dims[mask[i]]
    if var in X:
      axis[i] = 0
      weight[i] = X[var]
    elif var in Y:
      axis[i] = 1
      weight[i] = Y[var]
    elif var in Z:
      axis[i] = 2
      weight[i] = Z[var]
    else:
      print "dim %s is not used" % var
      axis[i] = -1
      weight[i] = 0
  #print axis
  #print weight
  reduced_data = np.zeros(shape=(len(data), 3))
  i = 0
  for d in data:
    r = [0, 0, 0]
    j = 0
    if binary:
      for e in d:
        r[axis[j]] += weight[j] * e * mask_ssd[j]
        j += 1
    else:
      for e in d:
        r[axis[j]] += weight[j] * e
        j += 1
    reduced_data[i] = r
    i += 1
  #print len(reduced_data)
  reduced_data = preprocessing.scale(reduced_data)
  if projection_mode == 3:
      pca = PCA(n_components=n_components)
      reduced_data = pca.fit_transform(reduced_data)
  print "ok !"

## classifiers initialisation  
#mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=10000,
#                      n_init=5, max_no_improvement=10, verbose=0)
km = KMeans(init='k-means++', n_clusters=n_clusters, precompute_distances=False,  
            n_init=10,  verbose=0)
dbs = DBSCAN(eps=0.3, min_samples=20)#,metric=metric)
#dbs1M = DBSCAN(eps=0.3, min_samples=100)#,metric=metric)
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
if (projection_mode == 1 or projection_mode == 2): #pca or manual
  print "on projected data"
  clf.fit(reduced_data)
else:
  clf.fit(data)
print 'ok : %s s' % (time.time()-t0)
print "all good .."

## writing classifier for whole dataset prediciton
with open(get_filename("clf.pk"),"wb") as filehandler:
  pickle.dump(clf, filehandler)

## retrieving infos for display, clusters stats ...
labels = clf.labels_
labels = [x + 1 for x in labels]
got_centers = False
try:
  cluster_centers = clf.cluster_centers_
  got_centers = True
except:
  print 'cannot have centers of clusters for algo : %s' % choose_clf
labels_unique = np.unique(labels)

## file loading for pandas
if compute_pandas:
  print "### using pandas ###"
  data_pd = []
  if binary:
    data_pd =  pd.io.parsers.read_csv(filename_clf, header=None)#, names=['Gender','Age','CSP','Geo','I0','I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14','I15','I16','MI0','MI1','MI2','MI3','MI4','MI5','MI6','MI7','MI8'])
    data_pd.columns = idx
  else:
    data_pd =  pd.io.parsers.read_csv(filename_pd, header=None, names=['Gender','Age','CSP','Geo','I0','I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','I14','I15','I16','MI0','MI1','MI2','MI3','MI4','MI5','MI6','MI7','MI8'])
  l = pd.Series(labels, name='labels')
  l2 = pd.DataFrame(l)
  data_pandas = pd.concat([data_pd, l2], axis = 1)

## counting ..
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
    filters_per_label[int(labels[i])][e][data_ini[i][e]] += 1
print 'without pandas : %s s' % (time.time()-t0)
if compute_pandas:
  res = dict()
  t0 = time.time()
  for t in data_pandas.axes[1][:-1]:
    res[t] = data_pandas.groupby(['labels',t]).count()

  print 'with pandas : %s s' % (time.time()-t0)

##print "combinaisons : %s " % len(count_combi)
 #sorted_count = sorted(count_combi.iteritems(), key=operator.itemgetter(1), reverse = True)
 #for i in range(100):
##  print sorted_count[i]
first_write_interests = True

def get_label(indice):
  return "%s#%s" % (dims[mask[indice]], mask_ssd[indice])

def to_label(dim, ss_dim_id):
  return "%s#%s" % (dim, ss_dim_id)

def interest_drop(d_label, current_path, w):
  global first_write_interests
  for k, ss_dim in ssdims[dims_ids['interests']].items():
    l = len(d_label[d_label[to_label('interests',k)] == 1])
    if l > 0:
      if first_write_interests:
        first_write_interests = False
        w.write('{"p1":"%s", "p2":"%s", "interest":"%s", "count":%s}' % (current_path[0], current_path[1], ss_dim, l))
      else:
        w.write(',{"p1":"%s", "p2":"%s", "interest":"%s", "count":%s}' % (current_path[0], current_path[1], ss_dim, l))

first_write_mi = True 

def market_intent_drop(d_label, current_path, w):
  global first_write_mi
  for k, ss_dim in ssdims[dims_ids['market_intent']].items():
    l = len(d_label[d_label[to_label('market_intent',k)] == 1])
    if l > 0:
      if first_write_mi:
        first_write_mi = False
        w.write('{"p1":"%s", "p2":"%s", "mi":"%s", "count":%s}' % (current_path[0], current_path[1], ss_dim, l))
      else:
        w.write(',{"p1":"%s", "p2":"%s", "mi":"%s", "count":%s}' % (current_path[0], current_path[1], ss_dim, l))
## may be a problem if more than max_depth filters are > threshold (male-urban-student -> only have 18- left ..) 

def recursive_drop(d_label, dims_to_consider, current_path, w_cluster, w_interest, w_mi):
  if (len(dims_to_consider) > 0 and len(current_path) < max_depth ):
    d_id, d = dims_to_consider[0]
    for a,ss_dim in ssdims[d_id].items():
      lab = to_label(d,a)
      reduced_d_label = d_label[d_label[lab] == 1]
      new_path = copy.deepcopy(current_path)
      new_path.append(lab)
      recursive_drop(reduced_d_label, dims_to_consider[1:], new_path, w_cluster, w_interest, w_mi)
  else:
    #path = "#".join(current_path)
    # write "csp#0|geography#1"
    #path = "|".join(map(lambda x:"#".join([str(y) for y in reverse_dims[x]]), current_path))
    path = "|".join(current_path)
    if len(d_label) > 0:
      w_cluster.writerow([path, len(d_label)])
      interest_drop(d_label, current_path, w_interest)
      market_intent_drop(d_label, current_path, w_mi)

def create_cluster(d_label, dims_to_consider, label_id, infos, filter_fixed): 
  print dims_to_consider
  print "filter fixed : ", filter_fixed 
  global first_write_mi, first_write_interests
  first_write_interests = True
  first_write_mi = True
  filename_interest = "data_clustering/" + get_filename("interests-%s.data" % p)
  w_interest = open(filename_interest, 'wb')
  w_interest.write('{"data": [')
  filename_mi = "data_clustering/" + get_filename("mis-%s.data" % p)
  w_mi = open(filename_mi, 'wb')
  w_mi.write('{"data": [')

  filename_cluster = "data_clustering/" + get_filename("cluster-%s.csv" % label_id)
  infos["names"].append((label_id, filename_cluster))
  infos["interests"].append((label_id, filename_interest))
  infos["marketIntents"].append((label_id, filename_mi))
  infos["counts"].append((label_id, len(d_label)))
  if (len(filter_fixed) == 1):
    infos["filters"].append((label_id, filter_fixed[0]))
    #infos["filters"].append((label_id, "#".join([str(x) for x in reverse_dims[filter_fixed[0]]])))
  elif (len(filter_fixed) >= 2):
    infos["filters"].append((label_id, "%s|%s" % (filter_fixed[0], filter_fixed[1])))
    #infos["filters"].append((label_id, ("#".join([str(x) for x in reverse_dims[filter_fixed[0]]]) + "|" + ("#".join([str(x) for x in reverse_dims[filter_fixed[1]]])))))
  else:
    infos["filters"].append((label_id, "No Filters"))
  w_cluster = csv.writer(open(filename_cluster, 'wb'))
  print "dims to consider : ", dims_to_consider 
  recursive_drop(d_label, dims_to_consider, [], w_cluster, w_interest, w_mi)

  w_interest.write(']}')
  w_mi.write(']}')

def write_cluster_info(infos):
  w_info = open(get_filename("cluster_info.json"), 'wb')
  w_info.write('{"nb" : %s' % n_clusters)
  for key, list_cl in infos.items():
    if key != "nb":
      w_info.write(',"%s" : [' % key )
      first = True
      for k, v in list_cl:
        if (first == False):
          w_info.write(',')
        w_info.write('{"%s" : "%s"}' % (k,v))
        first = False
      w_info.write(']')
  w_info.write('}')


infos = dict()
infos["names"] = []
infos["interests"] = []
infos["marketIntents"] = []
infos["counts"] = []
infos["filters"] = []

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
      e = {}
      if compute_pandas:
        e_t = []
        #if ssdims[mask[i]][mask_ssd[i]] == 'gender#0':
        if i == 0:
          e_t = res[get_label(i)][get_label(1)] #use ['Female'] to get first column only as we are using 'count' and all rows are equal
        else:
          e_t = res[get_label(i)][get_label(0)] #use ['Female'] to get first column only as we are using 'count' and all rows are equal
        try: 
          e[0] = e_t[p,False]
        except:
          e[0] = 0
        try: 
          e[1] = e_t[p,True]
        except:
          e[1] = 0
      else:
        e = filters_per_label[p][i]
      if (e[1] + e[0] == 0):
        ldim[mask[i]].append([0, e[0], e[1], ssdims[mask[i]][mask_ssd[i]]])
      else:
        ldim[mask[i]].append([100*(float(e[1])/(e[0]+e[1])), e[0], e[1], ssdims[mask[i]][mask_ssd[i]], i])
    for d, name in sorted_dims:
      ldim[d].sort()
      ldim[d].reverse()
      try:
        bdim[d] = ldim[d][0][0]
      except:
        print 'no data, skipping'
    best_dims = sorted(bdim.iteritems(), key=operator.itemgetter(1), reverse = True)
    #best_dims = []
    trunc_dim = copy.deepcopy(always_bin) # retrieve Intersets, MI, Funnel..
    #print "tempo : ", best_dims_t
    #for e in best_dims_t:
    #  if dims[e[0]] not in trunc_dim:
    #    best_dims.append(e)
    print "best dims : ", best_dims
    filter_real_count = []
    for d,s in best_dims:
      print dims[d]
      for ele in ldim[d][:10]:
        if (ele[0] > all_in_threshold) and (dims[d] not in trunc_dim):
          filter_real_count.append(ele[4])
        print "\t%s : %s in, %s out ( %.2f %%)" % (ele[3], ele[2], ele[1], ele[0])
      print "\t\t__ __ __"
    d_label = []
    filter_fixed = []
    if (len(filter_real_count) == 1):
      indice = filter_real_count[0]
      #indice_u = (ssdims[mask[indice]][mask_ssd[indice]])
      #indice_u = "%s_%s" % (dims[mask[indice]], mask_ssd[indice])
      indice_u = get_label(indice)
      trunc_dim.add(dims[mask[indice]])
      d_label = data_pandas[(data_pandas[indice_u] == 1) & (data_pandas['labels'] == p)]
      print "\n\tCombinaison of one %s %% filter : %s (%s)" % (all_in_threshold, len(d_label), indice_u)
      filter_fixed.append(indice_u)

    elif (len(filter_real_count) >= 2):
      indice0 = filter_real_count[0]
      indice1 = filter_real_count[1]
      #indice0_u = ssdims[mask[indice0]][mask_ssd[indice0]]
      #indice1_u = ssdims[mask[indice1]][mask_ssd[indice1]]
      #print dims[mask[indice0]]
      #print dims[mask[indice1]]
      #indice0_u = "%s_%s" % (dims[mask[indice0]], mask_ssd[indice0])
      #indice1_u = "%s_%s" % (dims[mask[indice1]], mask_ssd[indice1])
      indice0_u = get_label(indice0)
      indice1_u = get_label(indice1)
      trunc_dim.add(dims[mask[indice0]])
      trunc_dim.add(dims[mask[indice1]])
      d_label = data_pandas[(data_pandas[indice0_u] == 1) & (data_pandas[indice1_u] == 1) & (data_pandas['labels'] == p)]
      print "\n\tCombinaison of two %s %% filters : %s (%s, %s)" % (all_in_threshold,
                                                                    len(d_label), 
                                                                    indice0_u, 
                                                                    indice1_u) 
      filter_fixed.append(indice0_u)
      filter_fixed.append(indice1_u)

    dims_to_consider = []
    for d_id, _ in best_dims:
      if dims[d_id] not in trunc_dim:
        dims_to_consider.append((d_id, dims[d_id]))
    create_cluster(d_label, dims_to_consider, p, infos, filter_fixed)
  
  # no binary projection __ IT MAY NOT WORK ANYMORE    
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

# write json which stores all path to data storage (used by front)
write_cluster_info(infos)

## compute if possible distance between clusters
if got_centers:
  center_distances = euclidean_distances(cluster_centers)
  print "\tlabel\tlabel\tdistance"
  for i in labels_unique:
    for j in labels_unique:
      if i < j:
        print "\t%s\t%s\t%.3f" % (i, j, center_distances[i-1][j-1])


###################
####  DISPLAY  ####
###################

if display:
  color={-1:'c',0:'r', 1:'b', 2:'g', 3:'y', 4:'k', 5:'m', 6:'c', 7:0.1, 8:0.3, 9:0.8}
  for i in xrange(10, 100):
    color[i]=random.random()
  
  def get_color(c):
    if c < 100:
      return color[c]
    else:
      return 0.5
  
  
  if (projection_mode == 1):
    print "pca variance ratio : %s " % ','.join([str(v) for v in pca.explained_variance_ratio_])
  #  print pca.components_
  if (n_components == 2 and projection_mode == 1):
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
  
  elif (n_components >= 3 and projection_mode >= 1):
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
          #ax.scatter(reduced_data_rand[:, i], reduced_data_rand[:, j], reduced_data_rand[:, k], c=[get_color(int(l)) for l in labels[:n_points]]) #, marker=[mar[l] for l in labels[:n_points]])
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
