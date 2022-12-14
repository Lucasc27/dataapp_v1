import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
import pandas as pd
import phik
import scipy.stats as ss
import itertools

class EDA:

    def __init__(self, dataframe):
        self.df = dataframe
        self.columns = self.df.columns
        self.num_vars = self.df.select_dtypes(include=[np.number]).columns
        self.cat_vars = self.df.select_dtypes(include=[np.object, 'category']).columns

    def box_plot(self, main_var, col_x=None, hue=None):
        return px.box(self.df, x=col_x, y=main_var, color=hue)

    def violin(self, main_var, col_x=None, hue=None, split=False):
        sns.set(style="whitegrid")
        return sns.violinplot(x=col_x, y=main_var, hue=hue,
                    data=self.df, palette="husl", split=split)

    def swarmplot(self, main_var, col_x=None, hue=None, split=False):
        sns.set(style="whitegrid")
        return sns.swarmplot(x=col_x, y=main_var, hue=hue,
                    data=self.df, palette="husl", dodge=split)
    
    def histogram_num(self, main_var, hue=None, bins = None, ranger=None):
        return px.histogram(self.df[self.df[main_var].between(left = ranger[0], right = ranger[1])], \
            x=main_var, nbins =bins , color=hue, marginal='rug')

    def scatter_plot(self, col_x,col_y,hue=None, size=None, hover_data=None):
        return px.scatter(self.df, x=col_x, y=col_y, color=hue,size=size, hover_data=hover_data)

    def bar_plot(self, col_y, col_x, hue=None):
        return px.bar(self.df, x=col_x, y=col_y,color=hue)
    #def bar_plot(self, col_y, hue=None):
    #    return px.bar(self.df, y=col_y,color=hue)
        
    def line_plot(self, col_y,col_x,hue=None, group=None):
        return px.line(self.df, x=col_x, y=col_y,color=hue, line_group=group)

    def CountPlot(self, main_var, hue=None):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        chart = sns.countplot(x=main_var, data=self.df, hue=hue, palette='pastel')
        return chart.set_xticklabels(chart.get_xticklabels(), rotation=30)

    def heatmap_vars(self,cols, func = np.mean):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        chart = sns.heatmap(self.df.pivot_table(index =cols[0], columns =cols[1],  values =cols[2], aggfunc=func, fill_value=0).dropna(axis=1), annot=True, annot_kws={"size": 7}, linewidths=.5)
        return chart.set_xticklabels(chart.get_xticklabels(), rotation=30)

    def Corr(self, cols=None, method = 'pearson', cols_list_rv=None):

        def cramers_corrected_stat(confusion_matrix):
            """ calculate Cramers V statistic for categorical-categorical association.
                uses correction from Bergsma and Wicher, 
                Journal of the Korean Statistical Society 42 (2013): 323-328
            """
            chi2 = ss.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2/n
            r,k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        if len(cols) != 0:
            if len(cols_list_rv) != 0:
                if method == 'phik':
                    corr = phik.phik_matrix(self.df[cols].drop(cols_list_rv,axis=1))
                elif method == 'cramer v':
                    cols = list(self.df[cols].drop(cols_list_rv,axis=1).columns)
                    corrM = np.zeros((len(cols),len(cols)))
                    # there's probably a nice pandas way to do this
                    for col1, col2 in itertools.combinations(cols, 2):
                        idx1, idx2 = cols.index(col1), cols.index(col2)
                        corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(self.df[col1], self.df[col2]))
                        corrM[idx2, idx1] = corrM[idx1, idx2]
                    corr = pd.DataFrame(corrM, index=cols, columns=cols)
                else:
                    corr = self.df[cols].drop(cols_list_rv,axis=1).corr(method = method)
            else:
                if method == 'phik':
                    corr = phik.phik_matrix(self.df[cols])
                elif method == 'cramer v':
                    cols = list(self.df[cols].columns)
                    corrM = np.zeros((len(cols),len(cols)))
                    # there's probably a nice pandas way to do this
                    for col1, col2 in itertools.combinations(cols, 2):
                        idx1, idx2 = cols.index(col1), cols.index(col2)
                        corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(self.df[col1], self.df[col2]))
                        corrM[idx2, idx1] = corrM[idx1, idx2]
                    corr = pd.DataFrame(corrM, index=cols, columns=cols)
                else:
                    corr = self.df[cols].corr(method = method)
        else:
            if len(cols_list_rv) != 0:
                if method == 'phik':
                    corr = phik.phik_matrix(self.df.drop(cols_list_rv,axis=1))
                elif method == 'cramer v':
                    cols = list(self.df.drop(cols_list_rv,axis=1).columns)
                    corrM = np.zeros((len(cols),len(cols)))
                    # there's probably a nice pandas way to do this
                    for col1, col2 in itertools.combinations(cols, 2):
                        idx1, idx2 = cols.index(col1), cols.index(col2)
                        corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(self.df[col1], self.df[col2]))
                        corrM[idx2, idx1] = corrM[idx1, idx2]
                    corr = pd.DataFrame(corrM, index=cols, columns=cols)
                else:
                    corr = self.df.drop(cols_list_rv,axis=1).corr(method = method)
            else:
                if method == 'phik':
                    corr = phik.phik_matrix(self.df)
                elif method == 'cramer v':
                    cols = list(self.df.columns)
                    corrM = np.zeros((len(cols),len(cols)))
                    # there's probably a nice pandas way to do this
                    for col1, col2 in itertools.combinations(cols, 2):
                        idx1, idx2 = cols.index(col1), cols.index(col2)
                        corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(self.df[col1], self.df[col2]))
                        corrM[idx2, idx1] = corrM[idx1, idx2]
                    corr = pd.DataFrame(corrM, index=cols, columns=cols)
                else:
                    corr = self.df.corr(method = method)

        chart = sns.heatmap(corr, annot=True, annot_kws={"size": 7}, linewidths=.5)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=60)
        chart.set_yticklabels(chart.get_yticklabels(), rotation=30)
        return chart
   
    def DistPlot(self, main_var):
        sns.set(style="whitegrid")
        return sns.histplot(self.df[main_var], color='c', kde=True)