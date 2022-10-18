import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score

class classificationsPlotting():

    # Função para criar confusion matrix
    def confusionMatrix(self, yTest, yPred, labelPositive = 'Yes', labelNegative = 'False', classError = True):

        labelPositive = 'Yes'
        labelNegative = 'No'
        labels = [labelPositive, labelNegative]

        yTrue = [labelPositive if t == 1 else labelNegative for t in yTest]
        yPred = [labelPositive if t >= 0.5 else labelNegative for t in yPred]

        # Convertendo arrays para o tipo de dado categórico.

        yTrue = pd.Categorical(values = yTrue, categories = labels)
        yPred = pd.Categorical(values = yPred, categories = labels)

        # Transformando arrays em Séries Temporais.

        yPred = pd.Series(data = yPred, name = 'Predicted')
        yTrue = pd.Series(data = yTrue, name = 'Actual')

        # Criando a Confusion Matrix.

        cm = pd.crosstab(index = yPred, columns = yTrue, dropna = False)

        # Calculando os erros das classes da Confusion Matrix.

        if classError:

            # Capturando cada um dos valores da Confusion Matrix.

            truePositve, falsePositive, falseNegative, trueNegative = np.array(cm).ravel()

            # Criando um DataFrame contendo os erros das classes.

            ce = pd.DataFrame (
                data = [
                    falsePositive / (truePositve + falsePositive),
                    1 - trueNegative / (trueNegative + falseNegative)
                ],
                columns = ['classError'],
                index   = labels
            )

            # Inserindo no DataFrame, as colunas da Confusion Matrix.

            for c in range(cm.shape[1] - 1, -1, -1):

                # Inserindo as colunas no DataFrame.

                ce.insert(loc = 0, column = labels[c], value = cm[labels[c]])

            # Atribuindo índices e colunas ao DataFrame.

            ce.index   = pd.Series(ce.index, name = 'Predicted')
            ce.columns = pd.Series(ce.columns, name = 'Actual')

            # Retornando a Confusion Matrix com o erro das classes.

            return ce

        # Retornando Confusion Matrix.

        return cm


    # Definindo uma função, para realizar a plotagem de Confusions Matrix.
    def plotConfusionMatrix(self, data, figsize = (6, 6), fontScale = 1.2, 
                            title = 'Confusion Matrix', xlabel = 'Actual', ylabel = 'Predicted'):

        labelPositive = 'Yes'
        labelNegative = 'No'
        labels = [labelPositive, labelNegative]

        # Definindo a área de plotagem e suas dimensões.

        _, ax = plt.subplots(figsize = figsize)

        # Definindo o tamanho da fonte utilizada no gráfico.

        sns.set(font_scale = fontScale)

        # Criando Heatmap para representar a Confusion Matrix.

        ax = sns.heatmap (
            data       = data,
            annot      = True,
            cmap       = 'Blues',
            linewidths = 5,
            cbar       = False,
            fmt        = 'd'
        ) 

        # Definindo as labels e o título do gráfico. 

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel) 
        ax.set_title(title)

        # Definindo as ticklabels do gráfico.

        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels);


    # Definindo uma função, para calcular as métricas baseadas na Confusion Matrix.
    def getClassificationMetrics(self, yTest, predProb, labelPositive = 'Yes', labelNegative = 'No'):

        # Binarizando os scores obtidos nas previsões.
        yTrue = [labelPositive if t == 1 else labelNegative for t in yTest]
        yPred = [labelPositive if v >= 0.5 else labelNegative for v in predProb]

        # Convertendo arrays para o tipo categórico.
        labels = [labelPositive, labelNegative]

        yTrue = pd.Categorical(values = yTrue, categories = labels)
        yPred = pd.Categorical(values = yPred, categories = labels)

        # Convertendo arrays para o tipo numérico. 
        yNTrue = [1 if v == labelPositive else 0 for v in yTrue]
        yNPred = [1 if v == labelPositive else 0 for v in yPred]

        # Transformando arrays em Séries Temporais.
        yPred = pd.Series(data = yPred, name = 'Predicted')
        yTrue = pd.Series(data = yTrue, name = 'Actual')

        # Criando a Confusion Matrix.
        cm = self.confusionMatrix(yTest, predProb, labelPositive = labelPositive, labelNegative = labelNegative, classError = False)

        # Capturando cada um dos valores da Confusion Matrix.
        truePositve, falsePositive, falseNegative, trueNegative = np.array(cm).ravel()

        # Calculando as métricas.
        accuracy     = accuracy_score(yTrue, yPred)
        kappa        = cohen_kappa_score(yTrue, yPred)
        sensitivity  = recall_score(yNTrue, yNPred)
        specificity  = trueNegative /(trueNegative + falsePositive)
        #prevalence   = (truePositve + falseNegative) / len(yTrue)
        #ppv          = (sensitivity * prevalence) /((sensitivity * prevalence) + ((1 - specificity) * (1 - prevalence)))
        #npv          = (specificity * (1 - prevalence)) / (((1 - sensitivity) * prevalence) + ((specificity) * (1 - prevalence)))
        precision    = precision_score(yNTrue, yNPred)
        f1           = f1_score(yNTrue, yNPred)
        rocAuc       = roc_auc_score(yNTrue, predProb)
        error        = 1 - accuracy_score(yTrue, yPred)
        #bAccuracy    = balanced_accuracy_score(yTrue, yPred)

        # Criando um DataFrame, com o resultado das métricas calculadas.

        metrics = pd.DataFrame([{
            'Accuracy'            : accuracy,     # Determina a precisão geral prevista do modelo.
            'Kappa'               : kappa,        # Determina o coeficiente de Kappa.
            'Recall'              : sensitivity,  # Determina a proporção de registros positivos que foram classificados
                                                  # pelo algoritmo  como positivos.
            'Specificity'         : specificity,  # Determina a proporção de registros negativos que foram classificados 
                                                  # pelo algoritmo como negativos.
            #'Pos Pred Value'      : ppv,          # Determina a porcentagem de positivos previstos que são realmente positivos.
            #'Neg Pred Value'      : npv,          # Determina a porcentagem de negativos previstos que são realmente negativos.
            'Precision'           : precision,    # Determina a proporção de classificações positivas, que realmente 
                                                  # são positivas.
            'F1'                  : f1,           # Determina a média Harmônica entre a precision e o recall do modelo.
            'ROC AUC'             : rocAuc,       # Determina a medida de separabilidade ROC. Ela indica o quanto o modelo é 
                                                  # capaz de distinguir  as classes.   
            'Error'               : error,        # Determina o erro do modelo em relação a sua acurácia.
            #'Balanced Accuracy'   : bAccuracy,    # Determina a acurácia do modelo balanceada pelos tamanhos das classes.
            'Positive Class'      : labelPositive # Define qual classe é a classe positiva.
        }], index = ['Metrics']).transpose()

        # Retornando o DataFrame, com as métricas obtidas.

        return metrics
        
        
        
        
        
        
        
        
        
        
        
        
    