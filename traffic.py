from usermodel import UserModel
from config import DEFAULT_EXPLORATION_RATE
from collections import defaultdict
import json
import random
es_perf = [{"ActiveViewRate" : 0.3, "ClickRate" : 0.3, "ConversionRate" : 0.3 }]
es_prod ={}

def traffic(audience_json, Ni, exploration_rate=DEFAULT_EXPLORATION_RATE):
    Ne = int(Ni * exploration_rate)
    Na = Ni - Ne
    audience = json.loads(audience_json)
    user_model = UserModel()
    audience_users = user_model.filter(audience, limit=Na)
    exploratory_users = user_model.exclude(audience, limit=Ne)

    if len(audience_users) == 0 and Ni != 0:
        raise ValueError("No users match this audience: %s" % repr(audience))
    if len(exploratory_users) == 0 and exploration_rate != 0:
        raise ValueError("No users exist outside of this audience: %s" % repr(audience))

    FIELDS = "Impressions, Active views, Clicks, Conversions".split(', ')
    ### with transaction:  # FIXME
    response = defaultdict(int)
    for N, users in [(Na, audience_users), (Ne, exploratory_users)]:
        for _ in xrange(N):
            # Pick a user, and update their event values
            u = random.choice(users) # FIXME - implement frequency capping
            for field, b_event in zip(FIELDS, u.impress()):
                response[field] += b_event
    ### End transaction ###
    return json.dumps(response)

if __name__ == "__main__":
    random.seed(0)
    Ni = int(1e5)  # Number of impressions
    audience = {'Gender': 'Male', 'Age': '22-25', 'CSP': 'Student'}
    
    traffic(audience, Ni, exploration_rate=DEFAULT_EXPLORATION_RATE)

