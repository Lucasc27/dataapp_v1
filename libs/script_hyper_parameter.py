import numpy as np

class run_parameters():

    def __init__(self, type_tunning):
        
        self.type_tunning = type_tunning

    def execute(self):
        
        if self.type_tunning == "GRID":

            list_parameters = {
                'max_depth': [2,3],
                'n_estimators':[50,100],
                'learning_rate' : np.arange(0, 0.1, 0.005)
            }
            
            #list_parameters = {
            #    'penalty': ['l1','l2'],
            #    'C' : [1.0, 2.0],
            #    'class_weight': [{0:0.5, 1:0.7}, {0:0.7, 1:0.9}]
            #}
            
            #list_parameters = {
            #    'n_neighbors': [5,8,10],
            #    'algorithm':['auto','ball_tree','kd_tree','brute']
            #}

            return list_parameters

        else:
            
            return "Erro!"



            