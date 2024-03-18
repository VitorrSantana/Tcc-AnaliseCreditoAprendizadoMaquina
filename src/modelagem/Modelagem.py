# Importar bibliotecas necessárias
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.cluster         import KMeans
from sklearn.metrics         import RocCurveDisplay, accuracy_score, classification_report,roc_auc_score

import matplotlib.pyplot as plt
import lightgbm          as lgb
import xgboost           as xgb
import optuna
import pickle 
import warnings
warnings.filterwarnings("ignore")

class Modelagem:
    
    def __init__(self,data,col_target,model= None,perct_test_size=0.3):
        self.X = data.drop(col_target,axis=1)
        self.y = data[col_target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=perct_test_size, random_state=1)
        self.model = model

    def set_model(self,name_model='random_forest',params=None):
        self.name_model_set = name_model
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
                self.metricas_modelo()
                self.evaluate_model(type_model)
            case 'nao_supervisionado':
                # self.evaluate_values_of_k()
                self.model.fit(self.X)
            case _              : print('Não foi encontrado tipo de modelo')

    def evaluate_model(self,type_model='supervisionado'):
        match type_model:
            case 'supervisionado': 
                # Calculando para o treino
                self.y_train_pred = self.model.predict_proba(self.X_train)[:, 1]
                auc_roc_train = roc_auc_score(self.y_train, self.y_train_pred)
                print('Reporte de Classificação para o Treino:')
                print(f'A AUC ROC DO TREINO FOI >>> {auc_roc_train}')
                print(classification_report(self.y_train, self.model.predict(self.X_train)))
                RocCurveDisplay.from_predictions(self.y_train, self.y_train_pred)
                print('----------------------------------------')
                
                self.y_test_pred = self.model.predict_proba(self.X_test)[:, 1]
                print('Reporte de Classificação para o Teste:')
                auc_roc = roc_auc_score(self.y_test, self.y_test_pred)
                print(f'A AUC ROC DO TESTE FOI >>> {auc_roc}')
                print(classification_report(self.y_test, self.model.predict(self.X_test)))
                RocCurveDisplay.from_predictions(self.y_test, self.y_test_pred)
                print('----------------------------------------')
            case 'nao_supervisionado':
                print('constuir ainda')

    def metricas_modelo(self):
        
        self.auc_roc_train  = roc_auc_score(self.y_train, self.model.predict_proba(self.X_train)[:,1])
        self.accuracy_train = accuracy_score(self.y_train, self.model.predict(self.X_train))
        self.auc_roc_test   = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:,1])
        self.accuracy_test  = accuracy_score(self.ytest, self.model.predict(self.X_test))
        
        print(f'Treino-> AUC-ROC:{self.auc_roc_train} ---  ACCURACY {self.accuracy_train}')
        print(f'Treino-> AUC-ROC:{self.auc_roc_test} ---  ACCURACY {self.accuracy_test}')


    def apply_cross_validation(self,folds=5):
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=folds, scoring='roc_auc')  # 5-fold cross-validation
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

    def otimizacao_parametros_optuna(self,parametos_otimizar,num_iteracoes=100):
        def objective(trial, parametos_otimizar):
            # Define os hiperparâmetros a serem otimizados
            parametros = {}
            for param,tipo,valor_inicial,valor_final in parametos_otimizar:
                match tipo:
                    case 'float': parametros[param] = trial.suggest_float(param, valor_inicial,valor_final)
                    case 'int'  : parametros[param] = trial.suggest_int(param,valor_inicial,valor_final)
                    case 'str'  : parametros[param] = trial.suggest_categorical(param, [valor_inicial, valor_final])
                    case 'fixo' : parametros[param] = valor_inicial
                    case _      : print('Não foi encontrado coorespondencia') 
                
            parametros['random_state'] = 42
            print(parametros)

            # Usa o classificador definido da classe com os hiperparâmetros definidos para Otimizar
            self.model.set_params(**parametros)
            # model = self.model

            # Treina o modelo
            if self.name_model_set == 'xgboost':
                self.model.fit(self.X_train, self.y_train, verbose=True)
            else:
                self.model.fit(self.X_train, self.y_train)

            # Faz previsões no conjunto de validação
            # y_pred = self.model.predict_proba(self.X_test)[:, 1]
            self.metricas_modelo()
            # Calcula a métrica AUC-ROC
            # auc_roc = roc_auc_score(self.y_test, y_pred)
            # accuracy = accuracy_score(self.ytest, self.model.predict(self.X_test))
            # print(f'A AUC ROC DO TESTE FOI >>> {auc_roc}')
            # print(f'A ACCURACY  DO TESTE FOI >>> {auc_roc}')
            # Definir pesos para cada métrica (50% para AUC-ROC e 50% para acurácia)
            weight_auc_roc = 0.5
            weight_accuracy = 0.5
            
            # Calcular a pontuação composta como média ponderada das métricas
            score_composto = (weight_auc_roc * self.auc_roc_test) + (weight_accuracy * self.accuracy_test)

            # O objetivo é maximizar a métrica AUC-ROC com Accuracy
            return score_composto

        # Cria o estudo Optuna
        study = optuna.create_study(direction='maximize')

        # Parâmetros adicionais para a função objetivo
        # Use a função partial para passar os parâmetros adicionais para a função objetivo
        objective_with_params = lambda trial: objective(trial, parametos_otimizar)

        # Inicia a otimização
        study.optimize(objective_with_params, n_trials=num_iteracoes)
        melhores_param = study.best_params
        # Imprime os resultados
        for param,tipo,valor_inicial,valor_final in parametos_otimizar:
            match tipo:
                case 'fixo':  melhores_param[param] = valor_inicial
        
        self.model.set_params = melhores_param
        self.train_model()

        print('Melhor valor de SCORE', study.best_value)
        print('Melhores hiperparâmetros:', study.best_params)
        return study,study.best_params,study.best_value,self.auc_roc_train,self.accuracy_train,self.auc_roc_test,self.accuracy_test

