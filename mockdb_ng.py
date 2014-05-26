# -*- coding: utf-8 -*-
"""
Created on Tue May  6 11:07:36 2014

@author: william
"""

from csv import reader, writer
from collections import defaultdict 
from user import User
import random
from rule import Rule
from lbd_config.appconfig import CSV_PATH
from lbd.app.dao import *

def genCumDist(dist):  
  assert len(dist) > 0
  lst = [dist[0]]
  for i in dist[1:]:
    lst.append(lst[-1] + i)
  lst[-1] = 1
  return lst


def gen_non_bin_cat(label, cumDist):
  randval = random.random()
  for i, step in enumerate(cumDist):
    if randval <= step:
      return label[i]


def gen_bin_cat(dist):
  return random.random() < dist


def xuser(model, model_order, rules, coord):
  binary = False # change this if you want binary projection for ALL variables (True)
  multiple_val = set(['Interests','Market intent'])
  forbidden_val = set(['Funnel','Customer segmentation'])
  with open('dataset_1M_boostedinterest.csv','wb') as f:
    wri = writer(f)
    while True:
      user = User()
      for dim in model_order:
        if dim in ["Age", "Geography", "CSP", "Gender"]:
          dist = map(float, user.matches_rules(dim, rules).split(" "))
          cumDist = genCumDist(dist)
          user[dim] = gen_non_bin_cat(model[dim], cumDist)      
        elif dim in ["Interests", "Market intent"]:
          user[dim]=[]
          for ssdim in model[dim]:
            dist = float(user.matches_rules(ssdim, rules))
            if gen_bin_cat(dist):
              user[dim].append(ssdim) 
        elif dim == "Funnel":
          for ssdim in model[dim]:
            user[dim] = []
            dist = float(user.matches_rules(ssdim, rules))
            if ssdim in ["General population","Visitor"] and gen_bin_cat(dist):
              user[dim].append(ssdim) 
            elif ssdim == "Client" and "Visitor" in user[dim] and gen_bin_cat(dist):
              user[dim].append(ssdim)
            elif ssdim == "Loyal client" and "Client" in user[dim] and gen_bin_cat(dist):
              user[dim].append(ssdim)
        #elif dim == "Customer segmentation":
        #  if user["Funnel"] in ["Client","Loyal client"]:
        #    dist = map(float, user.matches_rules(dim, rules).split(" "))
        #    cumDist = genCumDist(dist)
        #    user[dim]= gen_non_bin_cat(model[dim], cumDist)
        #elif dim == "Client purchasing categories":
        #  if user["Funnel"] in ["Client","Loyal client"]:
        #    user[dim]=[]
        #    for ssdim in model[dim]:
        #      dist = float(user.matches_rules(ssdim, rules))
        #      if gen_bin_cat(dist): user[dim].append(ssdim) 
        #elif dim == "Performance":
        #  for perf in ['Active views',"Impressions", 'Clicks', 'Conversions']:
        #    user[perf] = 0
        #  for ssdim in model[dim]:
        #    dist = float(user.matches_rules(ssdim, rules))
        #    user.set_traffic_data(ssdim, dist)
        #else:
        #  raise Exception("Parse error: %s" % dim)
      if binary:
        u = [0] * (26 + 20)
      else:
        u = [0] * (26 + 4)
      for k,v in user.items():
        if binary:
          if k in multiple_val:
            for e in v:
              #print e, coord[k][e] + 4
              u[coord[k][e] + 20] = 1
          if k == 'Gender':
            u[coord[k][v]] = 1
          if k == 'Age':
            u[2 + coord[k][v]] = 1
          if k == 'CSP':
            u[9 + coord[k][v]] = 1 
          if k == 'Geography':
            u[16 + coord[k][v]] = 1
          
        else:
          if k in multiple_val:
            for e in v:
              #print e, coord[k][e] + 4
              u[coord[k][e] + 4] = 5
          if k == 'Gender':
            u[0] = coord[k][v]
          if k == 'Age':
            u[1] = coord[k][v]
          if k == 'CSP':
            u[2] = coord[k][v]
          if k == 'Geography':
            u[3] = coord[k][v]
      wri.writerow(u)
      yield user


def get_model_and_rules_from_csv():
  rules, model = defaultdict(list), defaultdict(list)
  model_order = () 
  coord = dict()
  seen_dim = set()
  with open('mdb.csv') as fh:
    sheet = reader(fh)
    for rowNb, row in enumerate(sheet):
      if (row[0] == "ordre croissant"):
        model_order = row[1:11]     
      if (row[0] == "regle"):
        dim1,crit1,dim2,crit2,dim3,crit3,dim_aff,value = row[1:9]
        rules[dim_aff].append(Rule(dim1, crit1, dim2, crit2, dim3, crit3, value))
      if rowNb < 67:
        dim, ssdim, num= row[12:15]
        model[dim] += [ssdim]
        if dim not in seen_dim:
          seen_dim.add(dim)
          coord[dim] = dict()
        coord[dim][ssdim] = int(num)
  return (model, model_order, rules, coord)

def get_user_gen():
  '''Returns a generator that yields users'''
  (model, model_order, rules, coord) = get_model_and_rules_from_csv()
  return xuser(model, model_order, rules, coord)
