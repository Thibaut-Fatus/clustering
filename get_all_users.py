import logging, time, csv
logging.basicConfig(level=logging.ERROR)

from elasticsearch import helpers, Elasticsearch
from app.mock_db_gen.user import User

from app.config.appconfig import USERS_INDEX, USERS_DOCTYPE, ES_NODES

import user_mask

import assign_all

#print ES_NODES
#print USERS_INDEX
#print USERS_DOCTYPE

es = Elasticsearch(ES_NODES)

request ='{"query":{"match_all" : {}}}'
scanResp = helpers.scan(client=es, query=request, scroll="10m", index=USERS_INDEX, doc_type=USERS_DOCTYPE, timeout="10m")

tempo_dict = {'gender':{0:0, 1:1}, 'age':{0:2,1:3,2:4,3:5,4:6,5:7,6:8}, 'csp':{0:9,1:10,2:11,3:12,4:13,5:14,6:15}, 'geography':{0:16,1:17,2:18,3:19}, 'interests':{0:20,1:21,2:22,3:23,4:24,5:25,6:26,7:27,8:28,9:29,10:30,11:31,12:32,13:33,14:34,15:35,16:36}, 'market_intent' : {0:37,1:38,2:39,3:40,4:41,5:42,6:43,7:44,8:45}}

forbidden = set(['funnel','client_purchasing_categories', 'customer_segmentation'])

i = 0
t0 = time.time()
with open('dump_from_db.csv', 'wb') as f:
  with open('labels.csv','wb') as fl:
    w = csv.writer(f)
    wl = csv.writer(fl)
    for r in scanResp:
      user = r["_source"]
      user_id = r["_id"]
      u = [0] * 46
      for k,v in user.items():
        if k not in forbidden:
          if type(v) == int:
            u[tempo_dict[k][v]] = 1
          else:
            for v2 in v:
              u[tempo_dict[k][v2]] = 1
      #print u
      label = assign_all.getLabel(u)
      w.writerow(u)
      wl.writerow([user_id,label[0]])
    
      i += 1
      if i%10000 == 0:
        t1 = time.time()
        print i, 10000/(t1-t0)
        t0 = t1

