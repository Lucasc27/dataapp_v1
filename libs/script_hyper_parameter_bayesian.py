from skopt.space import Real, Integer, Categorical

class run_parameters():

    def __init__(self, type_tunning):
        
        self.type_tunning = type_tunning

    def execute(self):
        
        if self.type_tunning == "Bayesian":

            list_parameters = [
             Integer(2,3, name='max_depth'),
             Integer(50,100, name='n_estimators'),
             Real(1e-3, 1e-1, 'log-uniform', name='learning_rate'),
             Integer(20, 50, name='num_leaves')
             #Categorical(['gbdt','dart'], name='boosting_type ')
            ]
            
            name_parameters = ['max_depth','n_estimators','learning_rate','num_leaves']

            return list_parameters, name_parameters

        else:
            
            return "Erro!"



            