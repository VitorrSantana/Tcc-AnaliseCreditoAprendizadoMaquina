# Importar bibliotecas necessárias
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.cluster         import KMeans
from sklearn.metrics         import RocCurveDisplay, accuracy_score, classification_report

import matplotlib.pyplot as plt
import lightgbm          as lgb
import xgboost           as xgb

class Modelagem:
    
    def __init__(self,data,col_target,model= None,perct_test_size=0.3):
        self.X = data.drop(col_target,axis=1)
        self.y = data[col_target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=perct_test_size, random_state=1)
        self.model = model


    def set_model(self,name_model='random_forest',params=None):
        match name_model:
            case 'random_forest': self.model = RandomForestClassifier(**params)
            case 'xgboost'      : self.model = xgb.XGBClassifier(**params)
            case 'reg_log'      : self.model = LogisticRegression(**params)
            case 'decis_tree'   : self.model = DecisionTreeClassifier(**params)
            case 'kmeans'       : self.model = KMeans(**params)
            case 'lgb'          : self.model = lgb.LGBMClassifier(**params)
            case _              : print('Não foi encontrado Modelo')

    def train_model(self,type_model='supervisionado'):
        match type_model:
            case 'supervisionado': 
                self.model.fit(self.X_train,self.y_train)
                self.evaluate_model(type_model)
            case 'nao_supervisionado':
                # self.evaluate_values_of_k()
                self.model.fit(self.X)
            case _              : print('Não foi encontrado tipo de modelo')

    def evaluate_model(self,type_model='supervisionado'):
        match type_model:
            case 'supervisionado': 
                # Calculando para o treino
                self.y_train_pred = self.model.predict(self.X_train)
                print('Reporte de Classificação para o Treino:')
                print(classification_report(self.y_train, self.y_train_pred))
                RocCurveDisplay.from_predictions(self.y_train, self.y_train_pred)
                print('----------------------------------------')
                
                self.y_test_pred = self.model.predict(self.X_test)
                print('Reporte de Classificação para o Teste:')
                print(classification_report(self.y_test, self.y_test_pred))
                RocCurveDisplay.from_predictions(self.y_test, self.y_test_pred)
                print('----------------------------------------')
            case 'nao_supervisionado':
                print('constuir ainda')

    def apply_cross_validation(self,folds=5):
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=folds, scoring='accuracy')  # 5-fold cross-validation
        print(f'Cross-Validation Scores: {cv_scores}')
        print(f'Mean Accuracy: {cv_scores.mean()}')

    def evaluate_values_of_k(self,last_value):
        inertia_values = []
        for k in range(1,last_value):
            self.set_model('kmeans',{'n_clusters':k})
            self.model.fit(self.X)
            inertia_values.append(self.model.inertia_)
        
        # Plotar o gráfico do método do cotovelo
        plt.plot(range(1, last_value), inertia_values, marker='o')
        plt.xlabel('Número de Clusters (K)')
        plt.ylabel('Inércia')
        plt.title('Método do Cotovelo para Escolha do Número de Clusters')
        plt.show()