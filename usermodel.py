
from decorators import limitable
import random


rand = lambda _: random.random()

class UserModel(list):
    '''A list of User dicts'''
    db = 0
    
    @limitable
    def filter_gen(self, audience, exclude=False):
        for u in sorted(self.db.users, key=rand):
            match = all([u.get(dim, None) == b  for dim, b in audience.items()])
            if match + exclude == 1:  # XOR
                yield u
                
    def filter(self, audience, limit=None):
        return list(self.filter_gen(audience, limit=limit))
            
    def exclude(self, audience, limit=None):
        return list(self.filter_gen(audience, limit=limit, exclude=True))
