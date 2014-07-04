## use this file ton configure how clustering is performed

## filename for classifier CAUTION use binary 
binary = True ##CAUTION need to be set in user_mask !!! 
#filename_clf = 'dataset_100k_boostedinterest.csv'
#filename_clf = 'dataset_1M_boostedinterest.csv'
filename_clf = 'dataset_100k_bin.csv'
#filename_clf = 'dataset_1M_bin.csv'
#filename_clf = 'dataset_20M_bin.csv'
#filename_clf = 'dump_from_db_1M.csv'

## filename for pandas stats
filename_pd = 'dataset_100k.csv'

## classifier relative
choose_clf = 'kmeans' # [kmeans, dbscan]
n_clusters = 8
scale_data = False # mean 0 and same std for all variables
val = 5 ## value given for weighted distance (interests / market intent)
projection_mode = 1 # (0 : none, 1 : PCA, 2 : Manual, 3 : Manuel + PCA)
n_components = 5 #PCA
n_points = 2000
compute_pandas = True
#metric=distance.euclidean
all_in_threshold = 90 # %, over which we only consider individuals who are in the combination of filters > threshold 

display = False
write_output = True
max_depth = 2 # for data visualisation and ID assignation (crossfilter)

compute_global_means = True # compute means of all indices, which gives us the repartition in % of ssdim (ex : 82% bargain hunter in overall pop) NEED TO BE BINARY

## projection interests X market intent X socio-demo 
X = {'Gender': 1, 'Age': 1, 'CSP' : 1, 'Geography' : 1}
#X = {'Gender': 0, 'Age': 0, 'CSP' : 1}
Y = {'Interests' : 1}
Z = {'Market intent' : 1}
#Z = {'Geography' : 1}

