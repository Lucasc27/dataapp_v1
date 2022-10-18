from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import os
from sklearn.linear_model import LogisticRegression
#from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import libs.metrics_ml as m_plt
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from matplotlib import pyplot as plt


class modelExperimentSetting():
    
    def __init__(self, have_base=None, base=None, target=None, split_sample=None, cols_combinations=None, test_size=None,
                models=None, mlflow_name_experiment=None, X_train=None, y_train=None, X_test=None, y_test=None):
        
        self.have_base = have_base
        self.base = base # Dataset
        self.X_train = X_train
        #self.y_train = y_train.values if have_base == None else y_train
        self.y_train = y_train[y_train.columns[0]]
        self.X_test = X_test
        #self.y_test = y_test.values if have_base == None else y_test
        self.y_test = y_test[y_test.columns[0]]
        self.target = target # Variável resposta
        self.split_sample = split_sample # Variável para particionar o dataset
        self.cols_combinations = cols_combinations # Variável com as combinações de features
        self.test_size = test_size # Variável para definir a divisão de dados de treino e teste
        self.models = models
        self.models_to_run = []
        self.mlflow_name_experiment = mlflow_name_experiment
        
        # Instanciando os modelos
        for name_model in self.models:
            if name_model == 'LogisticRegression':
                self.models_to_run.append([LogisticRegression(random_state=42), "LogisticRegression", "sklearn"])
            elif name_model == 'XGBClassifier':
                self.models_to_run.append([XGBClassifier(random_state=42), "XGBClassifier", "XGBoost"])
            elif name_model == 'LGBMClassifier':
                self.models_to_run.append([LGBMClassifier(random_state=42), "LGBMClassifier", "LGBoost"])
                #self.models_to_run.append([LGBMClassifier(learning_rate=0.1, num_leaves=126, min_child_samples=97, subsample=0.4400873805187065, colsample_bytree=0.670104177273168, random_state=42), "LGBMClassifier", "LGBoost"])
            elif name_model == 'RandomForestClassifier':
                self.models_to_run.append([RandomForestClassifier(random_state=42), "RandomForestClassifier", "sklearn"])
            elif name_model == 'AdaBoostClassifier':
                self.models_to_run.append([AdaBoostClassifier(random_state=42), "AdaBoostClassifier", "sklearn"])
            elif name_model == 'GradientBoostingClassifier':
                self.models_to_run.append([GradientBoostingClassifier(random_state=42), "GradientBoostingClassifier", "sklearn"])
            elif name_model == 'LinearDiscriminantAnalysis':
                self.models_to_run.append([LinearDiscriminantAnalysis(), "LinearDiscriminantAnalysis", "sklearn"])
            elif name_model == 'GaussianNB':
                self.models_to_run.append([GaussianNB(), "GaussianNB", "sklearn"])
            elif name_model == 'DecisionTreeClassifier':
                self.models_to_run.append([DecisionTreeClassifier(random_state=42), "DecisionTreeClassifier", "sklearn"])
            elif name_model == 'KNeighborsClassifier':
                self.models_to_run.append([KNeighborsClassifier(), "KNeighborsClassifier", "sklearn"])
    
    def execute(self):
        
        name_experiment = self.mlflow_name_experiment
        experiment_id = mlflow.set_experiment(name_experiment)
        
        #if self.split_sample:
        #    base_reject, self.base = train_test_split(self.base, test_size=self.split_sample, random_state=42)
        #    self.base = self.base.reset_index(drop=True)

        clf_plt = m_plt.classificationsPlotting()

        if not self.cols_combinations:

            if self.have_base:
                dataset = self.base
                X = dataset.drop([self.target],axis=1)
                y = dataset[self.target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
            else:
                X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

            for model, name, type_ml_package in self.models_to_run:
                
                with mlflow.start_run(run_name=name):
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)

                    confusion_matrix = clf_plt.confusionMatrix(y_test, y_pred[:,1])
                    metrics = clf_plt.getClassificationMetrics(y_test, y_pred[:,1])

                    TP = confusion_matrix.iloc[0,0]
                    FP = confusion_matrix.iloc[1,1]
                    TN = confusion_matrix.iloc[0,1]
                    FN = confusion_matrix.iloc[1,0]
                    accuracy = round(metrics[metrics.index=='Accuracy']['Metrics'][0] * 100, 2) 
                    recall = round(metrics[metrics.index=='Recall']['Metrics'][0] * 100, 2)
                    specificity = round(metrics[metrics.index=='Specificity']['Metrics'][0] * 100, 2)
                    precision = round(metrics[metrics.index=='Precision']['Metrics'][0] * 100, 2)
                    f1 = round(metrics[metrics.index=='F1']['Metrics'][0] * 100, 2)
                    auc = round(metrics[metrics.index=='ROC AUC']['Metrics'][0] * 100, 2)
                    kappa = round(metrics[metrics.index=='Kappa']['Metrics'][0] * 100, 2)
                    
                    mlflow.log_metric("Accuracy", accuracy)
                    mlflow.log_metric("TP", TP)
                    mlflow.log_metric("FP", FP)
                    mlflow.log_metric("TN", TN)
                    mlflow.log_metric("FN", FN)
                    mlflow.log_metric("Recall", recall)
                    mlflow.log_metric("Specificity", specificity)
                    mlflow.log_metric("Precision", precision)
                    mlflow.log_metric("F1", f1)
                    mlflow.log_metric("AUC", auc)
                    mlflow.log_metric("Kappa", kappa)
                    mlflow.log_metric("Nº columns", X_train.shape[1])
                    mlflow.log_metric("X train", X_train.shape[0])
                    mlflow.log_metric("X test", X_test.shape[0])

                    list_cols = list(X_train.columns)
                    for i,col in enumerate(list_cols):
                        if i == len(list_cols)-1:
                            list_cols[i] = col
                        else:
                            list_cols[i] = col + ','
                    features = ''
                    for col in list_cols:
                        features += col
                    mlflow.log_param("Features", features)

                    y_pred_prob_class_1 = [pred.tolist() for pred in y_pred[:,1]]
                    for i,pred in enumerate(y_pred_prob_class_1):
                        if i == len(y_pred_prob_class_1)-1:
                            y_pred_prob_class_1[i] = str(pred)
                        else:
                            y_pred_prob_class_1[i] = str(pred) + ','
                    y_pred_prob_class_1_str = ''
                    for pred in y_pred_prob_class_1:
                        y_pred_prob_class_1_str += pred
                    mlflow.log_param("y_pred_prob_class_1", y_pred_prob_class_1_str)


                    y_pred_prob_class_0 = [pred.tolist() for pred in y_pred[:,0]]
                    for i,pred in enumerate(y_pred_prob_class_0):
                        if i == len(y_pred_prob_class_0)-1:
                            y_pred_prob_class_0[i] = str(pred)
                        else:
                            y_pred_prob_class_0[i] = str(pred) + ','
                    y_pred_prob_class_0_str = ''
                    for pred in y_pred_prob_class_0:
                        y_pred_prob_class_0_str += pred
                    mlflow.log_param("y_pred_prob_class_0", y_pred_prob_class_0_str)

                    list_y_test = y_test.values.tolist()
                    for i,pred in enumerate(list_y_test):
                        if i == len(list_y_test)-1:
                            list_y_test[i] = str(pred)
                        else:
                            list_y_test[i] = str(pred) + ','
                    y_test_str = ''
                    for pred in list_y_test:
                        y_test_str += str(pred)
                    mlflow.log_param("y_test", y_test_str)
                
                    mlflow.log_param("Target", self.target)

                    #if type_ml_package == "XGBoost":
                    #    mlflow.xgboost.autolog()

                    #if type_ml_package != "XGBoost":
                    #dict_parameters = model.get_params()
                    #for param, value in dict_parameters.items():
                    #    mlflow.log_param(param, value)
                    if type_ml_package != "XGBoost": 
                        mlflow.log_params(model.get_params())
                    #else:
                    #    mlflow.log_params(model.get_xgb_params())
                    #mlflow.log_dict(dictionary_columns, "columns.txt")

                mlflow.end_run()
                

        else:
                
            for combination in self.cols_combinations:
                
                if self.have_base:
                    dataset = self.base
                    X = dataset[combination]
                    y = dataset[self.target]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=92)
                else:
                    X_train, X_test, y_train, y_test = self.X_train[combination], self.X_test[combination], self.y_train, self.y_test
                
                for model, name, type_ml_package in self.models_to_run:
                    
                    #if type_ml_package == "XGBoost":
                    #    mlflow.xgboost.autolog()

                    with mlflow.start_run(run_name=name):

                        model.fit(X_train, y_train)
                        y_pred = model.predict_proba(X_test)

                        confusion_matrix = clf_plt.confusionMatrix(y_test, y_pred[:,1])
                        metrics = clf_plt.getClassificationMetrics(y_test, y_pred[:,1])

                        TP = confusion_matrix.iloc[0,0]
                        FP = confusion_matrix.iloc[1,1]
                        TN = confusion_matrix.iloc[0,1]
                        FN = confusion_matrix.iloc[1,0]
                        accuracy = round(metrics[metrics.index=='Accuracy']['Metrics'][0] * 100, 2) 
                        recall = round(metrics[metrics.index=='Recall']['Metrics'][0] * 100, 2)
                        specificity = round(metrics[metrics.index=='Specificity']['Metrics'][0] * 100, 2)
                        precision = round(metrics[metrics.index=='Precision']['Metrics'][0] * 100, 2)
                        f1 = round(metrics[metrics.index=='F1']['Metrics'][0] * 100, 2)
                        auc = round(metrics[metrics.index=='ROC AUC']['Metrics'][0] * 100, 2)
                        kappa = round(metrics[metrics.index=='Kappa']['Metrics'][0] * 100, 2)
                        
                        mlflow.log_metric("Accuracy", accuracy)
                        mlflow.log_metric("TP", TP)
                        mlflow.log_metric("FP", FP)
                        mlflow.log_metric("TN", TN)
                        mlflow.log_metric("FN", FN)
                        mlflow.log_metric("Recall", recall)
                        mlflow.log_metric("Specificity", specificity)
                        mlflow.log_metric("Precision", precision)
                        mlflow.log_metric("F1", f1)
                        mlflow.log_metric("AUC", auc)
                        mlflow.log_metric("Kappa", kappa)
                        mlflow.log_metric("Nº columns", X_train.shape[1])
                        mlflow.log_metric("X train", X_train.shape[0])
                        mlflow.log_metric("X test", X_test.shape[0])

                        list_cols = list(X_train.columns)
                        for i,col in enumerate(list_cols):
                            if i == len(list_cols)-1:
                                list_cols[i] = col
                            else:
                                list_cols[i] = col + ','
                        features = ''
                        for col in list_cols:
                            features += col
                        mlflow.log_param("Features", features)

                        y_pred_prob_class_1 = [pred.tolist() for pred in y_pred[:,1]]
                        for i,pred in enumerate(y_pred_prob_class_1):
                            if i == len(y_pred_prob_class_1)-1:
                                y_pred_prob_class_1[i] = str(pred)
                            else:
                                y_pred_prob_class_1[i] = str(pred) + ','
                        y_pred_prob_class_1_str = ''
                        for pred in y_pred_prob_class_1:
                            y_pred_prob_class_1_str += pred
                        mlflow.log_param("y_pred_prob_class_1", y_pred_prob_class_1_str)


                        y_pred_prob_class_0 = [pred.tolist() for pred in y_pred[:,0]]
                        for i,pred in enumerate(y_pred_prob_class_0):
                            if i == len(y_pred_prob_class_0)-1:
                                y_pred_prob_class_0[i] = str(pred)
                            else:
                                y_pred_prob_class_0[i] = str(pred) + ','
                        y_pred_prob_class_0_str = ''
                        for pred in y_pred_prob_class_0:
                            y_pred_prob_class_0_str += pred
                        mlflow.log_param("y_pred_prob_class_0", y_pred_prob_class_0_str)

                        list_y_test = y_test.values.tolist()
                        for i,pred in enumerate(list_y_test):
                            if i == len(list_y_test)-1:
                                list_y_test[i] = str(pred)
                            else:
                                list_y_test[i] = str(pred) + ','
                        y_test_str = ''
                        for pred in list_y_test:
                            y_test_str += str(pred)
                        mlflow.log_param("y_test", y_test_str)
                        
                        mlflow.log_param("Target", self.target)

                        #if type_ml_package != "XGBoost":
                        #dict_parameters = model.get_params()
                        #for param, value in dict_parameters.items():
                        #    mlflow.log_param(param, value)
                        if type_ml_package != "XGBoost": 
                            mlflow.log_params(model.get_params())
                        #mlflow.log_dict(dictionary_columns, "columns.txt")

                    mlflow.end_run()


class saveModels():
    
    def __init__(self, have_base=None, base=None, model_name = None, X_train=None, y_train=None, X_test=None, y_test=None,
                        columns=None, target=None, test_size=None, params=None, name_model_to_save=None):
        
        self.have_base = have_base
        self.base = base
        self.model_name = model_name
        self.X_train = X_train
        #self.y_train = y_train.values if have_base == None else y_train
        self.y_train = y_train[y_train.columns[0]]
        self.X_test = X_test
        #self.y_test = y_test.values if have_base == None else y_test
        self.y_test = y_test[y_train.columns[0]]
        self.columns = columns
        self.target = target
        self.test_size = test_size
        self.params = params
        self.name_model_to_save = name_model_to_save

        self.list_params = {}
        for k,v in self.params.items():
            if '.' in v:
                self.list_params[k] = float(v)
            elif v.isdigit():
                self.list_params[k] = int(v)
            elif '-' in v:
                if '.' in v:
                    self.list_params[k] = float(v)
                elif 'e-' in v:
                    self.list_params[k] = float(v)
                else:
                    self.list_params[k] = int(v)
            elif v == 'None':
                self.list_params[k] = None
            elif v == 'True':
                self.list_params[k] = True
            elif v == 'False':
                self.list_params[k] = False
            else:
                self.list_params[k] = v

        self.model = None
        if model_name == 'LogisticRegression':
            self.model = LogisticRegression(**self.list_params)
        elif model_name == 'XGBClassifier':
            self.model = XGBClassifier(**self.list_params)
        elif model_name == 'LGBMClassifier':
            self.model = LGBMClassifier(**self.list_params)
        elif model_name == 'RandomForestClassifier':
            self.model = RandomForestClassifier(**self.list_params)
        elif model_name == 'AdaBoostClassifier':
            self.model = AdaBoostClassifier(**self.list_params)
        elif model_name == 'GradientBoostingClassifier':
            self.model = GradientBoostingClassifier(**self.list_params)
        elif model_name == 'LinearDiscriminantAnalysis':
            self.model = LinearDiscriminantAnalysis(**self.list_params)
        elif model_name == 'GaussianNB':
            self.model = GaussianNB(**self.list_params)
        elif model_name == 'DecisionTreeClassifier':
            self.model = DecisionTreeClassifier(**self.list_params)
        elif model_name == 'KNeighborsClassifier':
            self.model = KNeighborsClassifier(**self.list_params)

    def save(self):

        if self.have_base:
            dataset = self.base
            X = dataset[self.columns]
            y = dataset[self.target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        else:
            X_train, X_test, y_train, y_test = self.X_train[self.columns], self.X_test[self.columns], self.y_train, self.y_test

        clf_plt = m_plt.classificationsPlotting()

        self.model.fit(X_train, y_train)

        # Salvando o modelo
        pickle.dump(self.model, open(f'saved_models/{self.name_model_to_save}.sav', 'wb'))

        #y_pred = self.model.predict_proba(X_test)[:,1]
        y_pred = self.model.predict_proba(X_test)
        y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred[:,1]]

        confusion_matrix = clf_plt.confusionMatrix(y_test, y_pred[:,1])

        metrics = clf_plt.getClassificationMetrics(y_test, y_pred[:,1])
        metrics.reset_index(inplace=True)
        metrics.columns = ['Metric','Value']
        metrics = metrics[metrics['Metric']!='Positive Class']

        return metrics, y_train, y_test, y_pred, y_pred_binary


class grid_search_tunning():
    
    def __init__(self, data, target, columns, estimator, param_grid, scoring, cv, data_split, mlflow_name_experiment):

        self.data = data
        self.target = target
        self.columns = columns
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.data_split = data_split
        self.mlflow_name_experiment = mlflow_name_experiment

    def execute_grid_search(self):

        def run_grid():
            self.X = self.data[self.columns]
            self.y = self.data[self.target]
        
            gs = GridSearchCV(estimator = self.estimator, param_grid = self.param_grid, 
                                scoring = self.scoring, cv = self.cv)

            gs.fit(self.X, self.y)

            self.model = gs.best_estimator_
            self.type_model = str(type(gs.best_estimator_))
            self.best_score = gs.best_score_
            self.best_params = gs.best_params_

        run_grid()

        #model = gs.best_estimator_

        #type_model = str(type(gs.best_estimator_))

        #if type_model == "<class 'xgboost.sklearn.XGBClassifier'>":
        #    mlflow.xgboost.autolog()

        name_experiment = self.mlflow_name_experiment
        experiment_id = mlflow.set_experiment(name_experiment)

        with mlflow.start_run(run_name="Tuned_"+self.type_model.split(' ')[1].split('.')[-1].replace("'>","")):

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.data_split, random_state=42)

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict_proba(X_test)
            y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred[:,1]]

            clf_plt = m_plt.classificationsPlotting()
            confusion_matrix = clf_plt.confusionMatrix(y_test, y_pred[:,1])
            metrics = clf_plt.getClassificationMetrics(y_test, y_pred[:,1])

            TP = confusion_matrix.iloc[0,0]
            FP = confusion_matrix.iloc[1,1]
            TN = confusion_matrix.iloc[0,1]
            FN = confusion_matrix.iloc[1,0]
            accuracy = round(metrics[metrics.index=='Accuracy']['Metrics'][0] * 100, 2) 
            recall = round(metrics[metrics.index=='Recall']['Metrics'][0] * 100, 2)
            specificity = round(metrics[metrics.index=='Specificity']['Metrics'][0] * 100, 2)
            precision = round(metrics[metrics.index=='Precision']['Metrics'][0] * 100, 2)
            f1 = round(metrics[metrics.index=='F1']['Metrics'][0] * 100, 2)
            auc = round(metrics[metrics.index=='ROC AUC']['Metrics'][0] * 100, 2)
            kappa = round(metrics[metrics.index=='Kappa']['Metrics'][0] * 100, 2)

            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("TP", TP)
            mlflow.log_metric("FP", FP)
            mlflow.log_metric("TN", TN)
            mlflow.log_metric("FN", FN)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("Specificity", specificity)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("F1", f1)
            mlflow.log_metric("AUC", auc)
            mlflow.log_metric("Kappa", kappa)
            mlflow.log_metric("Nº columns", X_train.shape[1])
            mlflow.log_metric("X train", X_train.shape[0])
            mlflow.log_metric("X test", X_test.shape[0])

            list_cols = list(X_train.columns)
            for i,col in enumerate(list_cols):
                if i == len(list_cols)-1:
                    list_cols[i] = col
                else:
                    list_cols[i] = col + ','
            features = ''
            for col in list_cols:
                features += col
            mlflow.log_param("Features", features)

            y_pred_prob_class_1 = [pred.tolist() for pred in y_pred[:,1]]
            for i,pred in enumerate(y_pred_prob_class_1):
                if i == len(y_pred_prob_class_1)-1:
                    y_pred_prob_class_1[i] = str(pred)
                else:
                    y_pred_prob_class_1[i] = str(pred) + ','
            y_pred_prob_class_1_str = ''
            for pred in y_pred_prob_class_1:
                y_pred_prob_class_1_str += pred
            mlflow.log_param("y_pred_prob_class_1", y_pred_prob_class_1_str)


            y_pred_prob_class_0 = [pred.tolist() for pred in y_pred[:,0]]
            for i,pred in enumerate(y_pred_prob_class_0):
                if i == len(y_pred_prob_class_0)-1:
                    y_pred_prob_class_0[i] = str(pred)
                else:
                    y_pred_prob_class_0[i] = str(pred) + ','
            y_pred_prob_class_0_str = ''
            for pred in y_pred_prob_class_0:
                y_pred_prob_class_0_str += pred
            mlflow.log_param("y_pred_prob_class_0", y_pred_prob_class_0_str)

            list_y_test = y_test.values.tolist()
            for i,pred in enumerate(list_y_test):
                if i == len(list_y_test)-1:
                    list_y_test[i] = str(pred)
                else:
                    list_y_test[i] = str(pred) + ','
            y_test_str = ''
            for pred in list_y_test:
                y_test_str += str(pred)
            mlflow.log_param("y_test", y_test_str)
            
            mlflow.log_param("Target", self.target)

            #if type_ml_package != "XGBoost":
            #dict_parameters = model.get_params()
            #for param, value in dict_parameters.items():
            #    mlflow.log_param(param, value)
            if self.type_model != "<class 'xgboost.sklearn.XGBClassifier'>": 
                mlflow.log_params(self.model.get_params())
            #mlflow.log_dict(dictionary_columns, "columns.txt")

            metrics_output = clf_plt.getClassificationMetrics(y_test, y_pred[:,1])
            metrics_output.reset_index(inplace=True)
            metrics_output.columns = ['Metric','Value']
            metrics_output = metrics_output[metrics_output['Metric']!='Positive Class']

        mlflow.end_run()
        return self.best_score, self.best_params, self.model, "Model saved in mlflow", metrics_output, y_test, y_pred, y_pred_binary


class random_search_tunning():
    
    def __init__(self, data, target, columns, estimator, param_grid, scoring, cv, n_iter, data_split, mlflow_name_experiment):

        self.data = data
        self.target = target
        self.columns = columns
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter
        self.data_split = data_split
        self.mlflow_name_experiment = mlflow_name_experiment

    def execute_random_search(self):

        def run_random():
            self.X = self.data[self.columns]
            self.y = self.data[self.target]
        
            #gs = GridSearchCV(estimator = self.estimator, param_grid = self.param_grid, 
            #                    scoring = self.scoring, cv = self.cv)

            gs = RandomizedSearchCV(estimator = self.estimator, param_distributions = self.param_grid, 
                                        n_iter = self.n_iter, scoring = self.scoring, cv = self.cv, random_state=42)

            gs.fit(self.X, self.y)

            self.model = gs.best_estimator_
            self.type_model = str(type(gs.best_estimator_))
            self.best_score = gs.best_score_
            self.best_params = gs.best_params_
            self.iterations = gs.n_iter

        run_random()

        #model = gs.best_estimator_

        #type_model = str(type(gs.best_estimator_))

        #if type_model == "<class 'xgboost.sklearn.XGBClassifier'>":
        #    mlflow.xgboost.autolog()

        name_experiment = self.mlflow_name_experiment
        experiment_id = mlflow.set_experiment(name_experiment)

        with mlflow.start_run(run_name="Tuned_"+self.type_model.split(' ')[1].split('.')[-1].replace("'>","")):

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.data_split, random_state=42)

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict_proba(X_test)
            y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred[:,1]]

            clf_plt = m_plt.classificationsPlotting()
            confusion_matrix = clf_plt.confusionMatrix(y_test, y_pred[:,1])
            metrics = clf_plt.getClassificationMetrics(y_test, y_pred[:,1])

            TP = confusion_matrix.iloc[0,0]
            FP = confusion_matrix.iloc[1,1]
            TN = confusion_matrix.iloc[0,1]
            FN = confusion_matrix.iloc[1,0]
            accuracy = round(metrics[metrics.index=='Accuracy']['Metrics'][0] * 100, 2) 
            recall = round(metrics[metrics.index=='Recall']['Metrics'][0] * 100, 2)
            specificity = round(metrics[metrics.index=='Specificity']['Metrics'][0] * 100, 2)
            precision = round(metrics[metrics.index=='Precision']['Metrics'][0] * 100, 2)
            f1 = round(metrics[metrics.index=='F1']['Metrics'][0] * 100, 2)
            auc = round(metrics[metrics.index=='ROC AUC']['Metrics'][0] * 100, 2)
            kappa = round(metrics[metrics.index=='Kappa']['Metrics'][0] * 100, 2)

            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("TP", TP)
            mlflow.log_metric("FP", FP)
            mlflow.log_metric("TN", TN)
            mlflow.log_metric("FN", FN)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("Specificity", specificity)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("F1", f1)
            mlflow.log_metric("AUC", auc)
            mlflow.log_metric("Kappa", kappa)
            mlflow.log_metric("Nº columns", X_train.shape[1])
            mlflow.log_metric("X train", X_train.shape[0])
            mlflow.log_metric("X test", X_test.shape[0])

            list_cols = list(X_train.columns)
            for i,col in enumerate(list_cols):
                if i == len(list_cols)-1:
                    list_cols[i] = col
                else:
                    list_cols[i] = col + ','
            features = ''
            for col in list_cols:
                features += col
            mlflow.log_param("Features", features)

            y_pred_prob_class_1 = [pred.tolist() for pred in y_pred[:,1]]
            for i,pred in enumerate(y_pred_prob_class_1):
                if i == len(y_pred_prob_class_1)-1:
                    y_pred_prob_class_1[i] = str(pred)
                else:
                    y_pred_prob_class_1[i] = str(pred) + ','
            y_pred_prob_class_1_str = ''
            for pred in y_pred_prob_class_1:
                y_pred_prob_class_1_str += pred
            mlflow.log_param("y_pred_prob_class_1", y_pred_prob_class_1_str)


            y_pred_prob_class_0 = [pred.tolist() for pred in y_pred[:,0]]
            for i,pred in enumerate(y_pred_prob_class_0):
                if i == len(y_pred_prob_class_0)-1:
                    y_pred_prob_class_0[i] = str(pred)
                else:
                    y_pred_prob_class_0[i] = str(pred) + ','
            y_pred_prob_class_0_str = ''
            for pred in y_pred_prob_class_0:
                y_pred_prob_class_0_str += pred
            mlflow.log_param("y_pred_prob_class_0", y_pred_prob_class_0_str)

            list_y_test = y_test.values.tolist()
            for i,pred in enumerate(list_y_test):
                if i == len(list_y_test)-1:
                    list_y_test[i] = str(pred)
                else:
                    list_y_test[i] = str(pred) + ','
            y_test_str = ''
            for pred in list_y_test:
                y_test_str += str(pred)
            mlflow.log_param("y_test", y_test_str)
            
            mlflow.log_param("Target", self.target)

            #if type_ml_package != "XGBoost":
            #dict_parameters = model.get_params()
            #for param, value in dict_parameters.items():
            #    mlflow.log_param(param, value)
            if self.type_model != "<class 'xgboost.sklearn.XGBClassifier'>": 
                mlflow.log_params(self.model.get_params())
            #mlflow.log_dict(dictionary_columns, "columns.txt")

            metrics_output = clf_plt.getClassificationMetrics(y_test, y_pred[:,1])
            metrics_output.reset_index(inplace=True)
            metrics_output.columns = ['Metric','Value']
            metrics_output = metrics_output[metrics_output['Metric']!='Positive Class']

        mlflow.end_run()
        return self.best_score, self.best_params, self.model, "Model saved in mlflow", self.iterations, metrics_output, y_test, y_pred, y_pred_binary


class bayesian_tunning():
    
    def __init__(self, data, target, columns, estimator, space, name_parameters, scoring, data_split, set_calls, set_n_initial_points, mlflow_name_experiment):

        self.data = data
        self.target = target
        self.columns = columns
        self.estimator = estimator
        self.space = space
        self.name_parameters = name_parameters
        self.scoring = scoring
        self.data_split = data_split
        self.set_calls = set_calls
        self.set_n_initial_points = set_n_initial_points
        self.mlflow_name_experiment = mlflow_name_experiment

    def execute_bayesian(self):

        self.X = self.data[self.columns]
        self.y = self.data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.data_split, random_state=42)

        @use_named_args(self.space)
        def treinar_modelo(**params):
            
            #print('\n')
            #print(params)
            
            mdl = self.estimator.set_params(**params)
            mdl.fit(X_train, y_train)
            p = mdl.predict_proba(X_test)[:,1]
            y_pred = mdl.predict(X_test)
            
            if self.scoring == 'accuracy':
                return -accuracy_score(y_test, y_pred)
            elif self.scoring == 'f1':
                return -f1_score(y_test, p)
            elif self.scoring == 'precision':
                return -precision_score(y_test, p)
            elif self.scoring == 'roc_auc':
                return -roc_auc_score(y_test, p)
            elif self.scoring == 'recall':
                return -recall_score(y_test, p)
            elif self.scoring == 'neg_log_loss':
                return -log_loss(y_test, p)
            #return -recall_score(y_test, p, pos_label=1)
            #return -accuracy_score(y_test, p)

        resultados_gp = gp_minimize(treinar_modelo, self.space, random_state=42, verbose=0, n_calls=self.set_calls, n_initial_points=self.set_n_initial_points)

        bayesian_params = {self.name_parameters[i]:p for i,p in enumerate(resultados_gp.x)}

        self.model = self.estimator.set_params(**bayesian_params)
        self.best_score = abs(resultados_gp.fun)
        self.type_model = str(type(self.model))

        name_experiment = self.mlflow_name_experiment
        experiment_id = mlflow.set_experiment(name_experiment)

        with mlflow.start_run(run_name="Tuned_"+self.type_model.split(' ')[1].split('.')[-1].replace("'>","")):

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict_proba(X_test)
            y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred[:,1]]

            clf_plt = m_plt.classificationsPlotting()
            confusion_matrix = clf_plt.confusionMatrix(y_test, y_pred[:,1])
            metrics = clf_plt.getClassificationMetrics(y_test, y_pred[:,1])

            TP = confusion_matrix.iloc[0,0]
            FP = confusion_matrix.iloc[1,1]
            TN = confusion_matrix.iloc[0,1]
            FN = confusion_matrix.iloc[1,0]
            accuracy = round(metrics[metrics.index=='Accuracy']['Metrics'][0] * 100, 2) 
            recall = round(metrics[metrics.index=='Recall']['Metrics'][0] * 100, 2)
            specificity = round(metrics[metrics.index=='Specificity']['Metrics'][0] * 100, 2)
            precision = round(metrics[metrics.index=='Precision']['Metrics'][0] * 100, 2)
            f1 = round(metrics[metrics.index=='F1']['Metrics'][0] * 100, 2)
            auc = round(metrics[metrics.index=='ROC AUC']['Metrics'][0] * 100, 2)
            kappa = round(metrics[metrics.index=='Kappa']['Metrics'][0] * 100, 2)

            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("TP", TP)
            mlflow.log_metric("FP", FP)
            mlflow.log_metric("TN", TN)
            mlflow.log_metric("FN", FN)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("Specificity", specificity)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("F1", f1)
            mlflow.log_metric("AUC", auc)
            mlflow.log_metric("Kappa", kappa)
            mlflow.log_metric("Nº columns", X_train.shape[1])
            mlflow.log_metric("X train", X_train.shape[0])
            mlflow.log_metric("X test", X_test.shape[0])

            list_cols = list(X_train.columns)
            for i,col in enumerate(list_cols):
                if i == len(list_cols)-1:
                    list_cols[i] = col
                else:
                    list_cols[i] = col + ','
            features = ''
            for col in list_cols:
                features += col
            mlflow.log_param("Features", features)

            y_pred_prob_class_1 = [pred.tolist() for pred in y_pred[:,1]]
            for i,pred in enumerate(y_pred_prob_class_1):
                if i == len(y_pred_prob_class_1)-1:
                    y_pred_prob_class_1[i] = str(pred)
                else:
                    y_pred_prob_class_1[i] = str(pred) + ','
            y_pred_prob_class_1_str = ''
            for pred in y_pred_prob_class_1:
                y_pred_prob_class_1_str += pred
            mlflow.log_param("y_pred_prob_class_1", y_pred_prob_class_1_str)


            y_pred_prob_class_0 = [pred.tolist() for pred in y_pred[:,0]]
            for i,pred in enumerate(y_pred_prob_class_0):
                if i == len(y_pred_prob_class_0)-1:
                    y_pred_prob_class_0[i] = str(pred)
                else:
                    y_pred_prob_class_0[i] = str(pred) + ','
            y_pred_prob_class_0_str = ''
            for pred in y_pred_prob_class_0:
                y_pred_prob_class_0_str += pred
            mlflow.log_param("y_pred_prob_class_0", y_pred_prob_class_0_str)

            list_y_test = y_test.values.tolist()
            for i,pred in enumerate(list_y_test):
                if i == len(list_y_test)-1:
                    list_y_test[i] = str(pred)
                else:
                    list_y_test[i] = str(pred) + ','
            y_test_str = ''
            for pred in list_y_test:
                y_test_str += str(pred)
            mlflow.log_param("y_test", y_test_str)
            
            mlflow.log_param("Target", self.target)

            #if type_ml_package != "XGBoost":
            #dict_parameters = model.get_params()
            #for param, value in dict_parameters.items():
            #    mlflow.log_param(param, value)
            if self.type_model != "<class 'xgboost.sklearn.XGBClassifier'>": 
                mlflow.log_params(self.model.get_params())
            #mlflow.log_dict(dictionary_columns, "columns.txt")

            metrics_output = clf_plt.getClassificationMetrics(y_test, y_pred[:,1])
            metrics_output.reset_index(inplace=True)
            metrics_output.columns = ['Metric','Value']
            metrics_output = metrics_output[metrics_output['Metric']!='Positive Class']

        mlflow.end_run()
        
        plot_convergence(resultados_gp)
        plt.savefig('files_output/result_bayesian.png')

        return self.best_score, bayesian_params, self.model, "Model saved in mlflow", metrics_output, y_test, y_pred, y_pred_binary

            
            
            
            
            
        
        

        
        
        
        
        
        
        
        
        
        
        