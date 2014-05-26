class Rule(object):
    
    def __init__(self,dim1,crit1,dim2,crit2,dim3,crit3,dist):
        self.criterions = { dim1:crit1, dim2:crit2, dim3:crit3 }
        self.dist = dist
    
    def applies(self,user):
        if set(self.criterions.keys()) == set([""]):
            return True
        for key in self.criterions.keys():
            if key == "":
                pass                      
            elif user[key] != self.criterions[key]:
                return False
        return True 