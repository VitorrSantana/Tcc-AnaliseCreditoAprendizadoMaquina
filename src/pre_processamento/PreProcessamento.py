import pandas as pd
import numpy  as np

from sklearn.feature_selection import mutual_info_classif, chi2,SelectKBest, f_classif
from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import OneHotEncoder,StandardScaler

class PreProcessamento:
    def __init__(self,data,columns=None,option='OneHotEncoder'):
        self.data  = data
        self.option = option
        if columns == None:
            self.columns = self.get_columns_objetct()
        self.method = None
        if option=='OneHotEncoder':
            self.modelo = OneHotEncoder(handle_unknown='ignore')

    def apply_method_encoding(self):
        dataframe_encode = pd.DataFrame(self.modelo.fit_transform(self.data[self.columns]).toarray(),columns = self.modelo.get_feature_names_out())

        return pd.concat([self.data,dataframe_encode],axis=1)

    def get_columns_objetct(self):
        return list(self.data.select_dtypes(include=[object]))
    

    def padronizacao_dados(self,data,col_identifier=['SK_ID_CURR','TARGET']):
        # Melhorar ainda
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data.drop(col_identifier,axis=1))
        X_scaled = pd.DataFrame(X_scaled,columns=data.drop(col_identifier,axis=1).columns) 
        
        for idx in range(len(col_identifier)):
            X_scaled.insert(idx,col_identifier[idx],data[col_identifier[idx]])
        
        return X_scaled 

    def check_outliers(self,data, lista_cols):

        for col in lista_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1    #IQR is interquartile range. 

            filter = (data[col] >= Q1 - (1.5 * IQR)) & (data[col] <= Q3 + (1.5 *IQR))
            print(col, len(data.loc[filter]))
    
    def mutual_information(self,X,y):
        # Calcula a pontuação de informação mútua para cada feature
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
        mutual_info_scores = mutual_info_classif(X_train, y_train)
        features_names = list(X_train.columns)
        features_selecionadas = sorted(zip(mutual_info_scores, features_names), key=lambda x: x[0], reverse=True)
        features_selecionadas = [feature[1] for feature in  features_selecionadas]
     
        return features_selecionadas
    
    def teste_f_classif(self,X,y,qtd_features_selecionadas=10):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

        # features_names = list(X.columns)
        # Usando o SelectKBest com o teste de Anova para selecionar as duas melhores características
        anova_score, _ = f_classif(X_train, y_train)
        features_names = list(X_train.columns)
        features_selecionadas = sorted(zip(anova_score, features_names), key = lambda x: x[0], reverse=True)
        features_selecionadas = [x[1] for x in features_selecionadas]

        return features_selecionadas

    def correlacao_target(self,X,y):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

        corr_ranking = X_train.corrwith(y_train, method = 'pearson')
        corr_ranking = corr_ranking.abs()
        corr_ranking.sort_values(ascending = False, inplace = True)
        corr_ranking = corr_ranking.index

        return list(corr_ranking)