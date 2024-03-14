import shap
from sklearn.model_selection import train_test_split

shap.initjs()

class Explicabilidade_Modelos:

    def __init__(self,model,type_model='tree') -> None:
        
        self.model = model
        if type_model == 'tree':
            self.explainer = shap.TreeExplainer(model)
        else: # mudar para adicionar mais modalidades
            pass

    
    def set_data(self,X,y=None,tipo_dado='Treino',porc_avaliada = 0.3,):

        match tipo_dado:
            case 'Treino': 
                self.X_Train, _, _, _  = train_test_split(X,y, train_size=porc_avaliada, random_state=1)
                self.shap_values_train = self.explainer.shap_values(self.X_Train)   
            case 'Teste' : 
                _, self.X_Test, _, _   = train_test_split(X,y, test_size=porc_avaliada, random_state=1)
                self.shap_values_test  = self.explainer.shap_values(self.X_Test) 
            case _       : 
                self.X                = self.X
                self.shap_values      = self.explainer.shap_values(self.X) 
        
    def grafico_force_plot_instace_unique(self,tipo_dado='Treino',col_identificacao = 'SK_ID_CURR',id_cliente=None):
        
        if id_cliente is None:
            idx_cliente = 0
        else:
            idx_cliente = self.X.query(f'{col_identificacao}=={id_cliente}').index
        
        shap.force_plot(self.explainer.expected_value[1],self.shap_values[1][idx_cliente,:],self.X.iloc[idx_cliente,:])


    def graficos_force_plot_instabces(self):
        pass     

    def grafico_summary_plot(self,tipo_dado='Treino'):
         match tipo_dado:
            case 'Treino':
                shap.summary_plot(self.shap_values_train[1],self.X_Train)
            case 'Teste':
                shap.summary_plot(self.shap_values_test[1],self.X_Test)
            case _:
                 shap.summary_plot(self.shap_values[1],self.X)


    def dependence_plot(self,feature_dependence,tipo_dado='Treino',interaction_index = None):

         match tipo_dado:
            case 'Treino':
                shap.dependence_plot(feature_dependence,self.shap_values_train[1],self.X_Train,interaction_index=interaction_index)
            case 'Teste':
                shap.dependence_plot(self.shap_values_test[1],self.X_Test,interaction_index=interaction_index)
            case _:
                 shap.dependence_plot(self.shap_values[1],self.X,interaction_index=interaction_index)

        