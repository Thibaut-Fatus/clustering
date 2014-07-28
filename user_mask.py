# coding=utf-8
import csv, operator
## loads dimensions : data sent to front need to be octopus compliant
from app.config.schema import dimensions_lst as dimensions

binary = True
## mapper to get dict of dimensions and subdimensions
print '### loading mapper ... ###'
dims = dict()
ssdims = dict()
dim_ids = set()
forbidden = set(["performance","performance_rate"])
for d in dimensions:
  if d['key'] not in forbidden:
    dim = d['key']
    dim_id = d['pos']
    for e in d['bins']:
      ssdim = e['name']
      ssdim_id = e['key']
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
always_bin = set(['market_intent', 'interests', 'funnel'])
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
for k,e in ssdims.items():
  for f,_ in e.items():
    idx.append("%s#%s" % (dims[k],f))

#inverse masks : first mask_pn indice -> dim_name, then dim_name -> dim_value (str) -> indice
mask_pn = {k:dims[v] for k,v in mask.items()}
mask_ssd_pn_inv = dict()
for k,v in mask_ssd.items():
  mask_ssd_pn_inv[mask_pn[k]] = dict()

for k,v in mask_ssd.items():
  mask_ssd_pn_inv[mask_pn[k]][ssdims[mask[k]][v]] = k

## retrive indices from dim pos and ssdim indice
indice_dict = dict()
i = 0
for k,v in dims.items():
 indice_dict[v] = dict()
 for j in range(len(ssdims[k])):
   indice_dict[v][j] = i
   i += 1

nb_ssd = i #we've counted the total nb of ss dims

