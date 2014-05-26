import csv, json, operator
from lbd.app.models import categories, cat_names

cat_ids = {v: k for k, v in cat_names.items()}
print cat_ids
INTERESTS = categories['interests'] + categories['online_purchase_habits']

cat_order = {"age" :0, "csp" : 1, "funnel" : 2, "gender" : 3 , "geography" : 4} ## removed : performance

## different from Baptiste's one
def format_user(user_dct):
  dct = {}
  usr_vect = [0 for i in range(len(cat_order) + len(INTERESTS))]
  for k, v in user_dct.items():
    if k in cat_order:
      #dct[k] = v
      usr_vect[cat_order[k]] = v
  for i, v in enumerate(INTERESTS):
    if user_dct[v]:
      usr_vect[i + len(cat_order)] = 1 
  #return dct, interests
  return usr_vect

def gen_header():
  sorted_cat = sorted(cat_order.iteritems(), key=operator.itemgetter(1))
  h = [x for x, y in sorted_cat] + INTERESTS
  print h
  return h

with open("dataset") as f:
  with open("dataset_clean",'wb') as fw:
    writer = csv.writer(fw)
    x = json.load(f)
    print "### dataset length : %s ###" % len(x)
    header = gen_header()
    writer.writerow(header)
    a = 0
    for u in x:
      usr_vect = format_user(u)
      writer.writerow(usr_vect)

