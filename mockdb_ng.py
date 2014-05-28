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
  mode = 2 # [0, 1, 2] 0 : binary, 1 : gender, age, csp & geo [0..1/7/7/4], 2 : write both files.
  val = 1 # value for interests / MI used when not projected
  multiple_val = set(['Interests','Market intent'])
  forbidden_val = set(['Funnel','Customer segmentation'])
  with open('dataset_1M.csv','wb') as f:
    wri = writer(f)
    with open('dataset_1M_bin.csv','wb') as f_b:
      wri_b = writer(f_b)
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
        if mode == 0:
          ub = [0] * (26 + 20)
          for k,v in user.items():
            if k in multiple_val:
              for e in v:
                #print e, coord[k][e] + 4
                ub[coord[k][e] + 20] = 1
            if k == 'Gender':
              ub[coord[k][v]] = 1
            if k == 'Age':
              ub[2 + coord[k][v]] = 1
            if k == 'CSP':
              ub[9 + coord[k][v]] = 1 
            if k == 'Geography':
              ub[16 + coord[k][v]] = 1
          wri_b.writerow(ub)
        elif mode == 2:
          ub = [0] * (26 + 20)
          u = [0] * (26 + 4)
          for k,v in user.items():
            if k in multiple_val:
              for e in v:
                #print e, coord[k][e] + 4
                ub[coord[k][e] + 20] = 1
                u[coord[k][e] + 4] = val
            if k == 'Gender':
              ub[coord[k][v]] = 1
              u[0] = coord[k][v]
            if k == 'Age':
              ub[2 + coord[k][v]] = 1
              u[1] = coord[k][v]
            if k == 'CSP':
              ub[9 + coord[k][v]] = 1 
              u[2] = coord[k][v]
            if k == 'Geography':
              ub[16 + coord[k][v]] = 1
              u[3] = coord[k][v]
          wri.writerow(u)
          wri_b.writerow(ub)
        elif mode == 1:
          u = [0] * (26 + 4)
          for k,v in user.items():
            if k in multiple_val:
              for e in v:
                #print e, coord[k][e] + 4
                u[coord[k][e] + 4] = val
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
