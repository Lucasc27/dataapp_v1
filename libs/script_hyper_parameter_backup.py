import numpy as np

class run_parameters():

    def __init__(self, type_tunning):
        
        self.type_tunning = type_tunning

    def execute(self):
        
        if self.type_tunning == "GRID":

            list_parameters = {
                'max_depth': [1,2,3,4,5],
                'n_estimators':[50,100],
                'learning_rate' : np.arange(0, 0.2, 0.005)
            }
            
            #list_parameters = {
            #    'penalty': ['l1','l2'],
            #    'C' : [1.0, 2.0],
            #    'class_weight': [{0:0.5, 1:0.7}, {0:0.7, 1:0.9}]
            #}

            return list_parameters

        else:
            
            return "Erro!"



            