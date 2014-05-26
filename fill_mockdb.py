import logging 
logging.basicConfig(level=logging.ERROR) 
import time, random
from mockdb import get_user_gen
from uuid import uuid4

from multiprocessing import Pool

def fill_mock_databases(count):
    '''
    Fills Elasticsearch databases
    '''
    t0 = time.time()
    gen = get_user_gen()
    for i in xrange(count):
        user = gen.next()
        try:
            #es.index(index='lbd', doc_type='user',id=uuid4(), body=user)

        except Exception as e:
            print e
            pass
        if i % 10000 == 0:
            t1 = time.time()
            if t0 > 0:
                print 10000 / ( t1 - t0 )
            t0 = t1

fill_mock_databases(200000)
