import random


FIELDS = ('Impressions', 'Active views', 'Conversions','Clicks')


class User(dict):

    def __init__(self, *a, **kw):
        self.activeViewRate = None 
        self.clickRate = None
        self.conversionRate = None
        self.model = {}
        super(User, self).__init__(*a, **kw)
    
    def matches_rules(self,dim, rules):
        res = [rule for rule in rules[dim] if rule.applies(self)]
        if len(res) != 0:
            return res[-1].dist
        raise ValueError("No rule on %s for %s" % (dim, repr(self)))

    def impress(self):
        assert all([f in self for f in FIELDS])
        '''Simulates an impression, with potential click and conversion'''
        self["Impressions"] += 1
        impression, activeview, click, conversion = True, False, False, False
        if random.random() <= self.activeViewRate:
            self['Active views'] += 1
            activeview = True
        if random.random() <= self.clickRate:
            self['Clicks'] += 1
            click = True
            if random.random() <= self.conversionRate:
                self['Conversions'] += 1
                conversion = True
        return impression, activeview, click, conversion
    
    def get_events(self):
        return self.impressions, self.activeviews, self.clicks, self.conversions

    def set_traffic_data(self,perf,rate):
        if perf == "ClickRate":
            self.clickRate = rate
        elif perf == "ActiveViewRate":
            self.activeViewRate = rate
        elif perf == "ConversionRate":
            self.conversionRate = rate
        else:
            raise Exception("Parse error: %s" % perf) 
                       
