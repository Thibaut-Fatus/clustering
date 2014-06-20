import csv, operator

binary = True
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

## inverse dims mapping to retrieve id from name
dims_ids = {v:k for k,v in dims.items()}
ssdims_ids = dict()
ssdims_ids_pn = dict()
for key, name in dims.items():
  ssdims_ids[key] = {v:k for k,v in ssdims[key].items()}
  ssdims_ids_pn[name] = {v:k for k,v in ssdims[key].items()}

## mask to get dims / subdims from indices
print '### creating mask ...'
sorted_dims = sorted(dims.iteritems(), key=operator.itemgetter(0))
mask = dict()
mask_ssd = dict()
i = 0
always_bin = set(['Market intent', 'Interests', 'Funnel'])
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

##list of ssdims (in order) for pandas
idx = []
for _,e in ssdims.items():
  for _,j in e.items():
    idx.append(j)

#inverse masks : first mask_pn indice -> dim_name, then dim_name -> dim_value (str) -> indice
mask_pn = {k:dims[v] for k,v in mask.items()}
mask_ssd_pn_inv = dict()
for k,v in mask_ssd.items():
  mask_ssd_pn_inv[mask_pn[k]] = dict()

for k,v in mask_ssd.items():
  mask_ssd_pn_inv[mask_pn[k]][ssdims[mask[k]][v]] = k

