# -*- coding: utf-8 -*-
"""
Created on Tue May  6 11:07:36 2014

@author: william
"""

from csv import reader
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


def xuser(model, model_order, rules):
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
            elif dim == "Customer segmentation":
                if user["Funnel"] in ["Client","Loyal client"]:
                    dist = map(float, user.matches_rules(dim, rules).split(" "))
                    cumDist = genCumDist(dist)
                    user[dim]= gen_non_bin_cat(model[dim], cumDist)
            elif dim == "Client purchasing categories":
                if user["Funnel"] in ["Client","Loyal client"]:
                    user[dim]=[]
                    for ssdim in model[dim]:
                        dist = float(user.matches_rules(ssdim, rules))
                        if gen_bin_cat(dist): user[dim].append(ssdim) 
            elif dim == "Performance":
                for perf in ['Active views',"Impressions", 'Clicks', 'Conversions']:
                    user[perf] = 0
                for ssdim in model[dim]:
                    dist = float(user.matches_rules(ssdim, rules))
                    user.set_traffic_data(ssdim, dist)
            else:
                raise Exception("Parse error: %s" % dim)
        yield user


def get_model_and_rules_from_csv():
    rules, model = defaultdict(list), defaultdict(list)
    model_order = () 
    with open(CSV_PATH) as fh:
        sheet = reader(fh)
        for rowNb, row in enumerate(sheet):
            if (row[0] == "ordre croissant"):
                model_order = row[1:11]         
            if (row[0] == "regle"):
                dim1,crit1,dim2,crit2,dim3,crit3,dim_aff,value = row[1:9]
                rules[dim_aff].append(Rule(dim1, crit1, dim2, crit2, dim3, crit3, value))
            if rowNb < 67:
                dim, ssdim = row[12:14]
                model[dim] += [ssdim]
    return (model, model_order, rules)

def get_user_gen():
    '''Returns a generator that yields users'''
    (model, model_order, rules) = get_model_and_rules_from_csv()
    return xuser(model, model_order, rules)
