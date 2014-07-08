# -*- coding: utf-8 -*-

from elasticsearch import helpers, Elasticsearch
from app.config.appconfig import USERS_INDEX, USERS_DOCTYPE, ES_NODES

import csv, time

es = Elasticsearch(ES_NODES)
n = 5000

def update_perf(actions):
    '''
    update users using bulk 
    '''
    res = helpers.bulk(client=es, actions=actions, chunk_size=n)
    return res

f = open('labels.csv','rb')
reader = csv.reader(f)

def commit():
  i = 1
  t0 = time.time()
  for uid, label in reader:
    if (i % n == 0):
      print i
      t1 = time.time()
      print n / (t1 - t0)
      t0 = t1
    i += 1
    body = {
      "doc": {
        "cluster": "%s" % label
        },
      "_op_type": "update",
      "_id": uid,
      "_index": USERS_INDEX,
      "_type": USERS_DOCTYPE
    }
    yield body

update_perf(commit())
